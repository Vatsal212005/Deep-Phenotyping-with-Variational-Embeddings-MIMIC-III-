#!/usr/bin/env python3
"""
MIMIC-III (clinical database demo v1.4) preprocessing
→ Produces a per-ICU-stay feature matrix suitable for VAE / clustering / anomaly detection.

- First 24h window (configurable)
- Unit hygiene (°F→°C), physiologic clipping
- Missingness indicators + RobustScaler
- Best-effort parquet (falls back to CSV if engine missing)
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from joblib import dump

# ---------------------------
# Config
# ---------------------------

@dataclass
class VarSpec:
    source: str  # 'chartevents' or 'labevents'
    patterns: List[str]  # regex patterns to match LABELs (case-insensitive)
    unit: str | None
    physiologic_min: float | None
    physiologic_max: float | None
    convert: str | None = None  # e.g., 'f_to_c'

VARS: Dict[str, VarSpec] = {
    # Vitals (CHARTEVENTS)
    "HR":   VarSpec("chartevents", [r"^heart\s*rate$", r"^heartrate$", r"^hr$"], "bpm",   20, 230, None),
    "SBP":  VarSpec("chartevents", [r"systolic.*blood.*pressure", r"sbp"],        "mmHg", 50, 250, None),
    "DBP":  VarSpec("chartevents", [r"diastolic.*blood.*pressure", r"dbp"],       "mmHg", 20, 150, None),
    "MAP":  VarSpec("chartevents", [r"mean arterial", r"map"],                     "mmHg", 30, 180, None),
    "RR":   VarSpec("chartevents", [r"respiratory\s*rate", r"^rr$"],               "bpm",   4,  80, None),
    "TEMP": VarSpec("chartevents", [r"temperature"],                               "C",   30.0, 43.0, "f_to_c"),
    "SPO2": VarSpec("chartevents", [r"spo2", r"o2\s*sat", r"oxygen.*saturation"], "%",    50, 100, None),
    # Labs (LABEVENTS)
    "WBC":   VarSpec("labevents", [r"^wbc$", r"white blood"], "K/µL",  0.5, 100.0, None),
    "HGB":   VarSpec("labevents", [r"^hgb$", r"hemoglobin"],  "g/dL",  2.0,  25.0, None),
    "PLT":   VarSpec("labevents", [r"platelet"],              "K/µL",  5.0, 1500,  None),
    "NA":    VarSpec("labevents", [r"^sodium$", r"^na$"],     "mmol/L",100,  180,  None),
    "K":     VarSpec("labevents", [r"^potassium$", r"^k$"],   "mmol/L",1.5,    9.0, None),
    "CL":    VarSpec("labevents", [r"^chloride$", r"^cl$"],   "mmol/L",60,   140,   None),
    "HCO3":  VarSpec("labevents", [r"bicarbonate", r"hco3"],  "mmol/L",5,     60,   None),
    "BUN":   VarSpec("labevents", [r"bun", r"urea"],          "mg/dL", 1,    300,   None),
    "CREAT": VarSpec("labevents", [r"creatinine"],            "mg/dL", 0.1,   25.0, None),
    "GLU":   VarSpec("labevents", [r"glucose"],               "mg/dL", 20,  1000,   None),
    "LACT":  VarSpec("labevents", [r"lactate"],               "mmol/L",0.2,   30.0, None),
    "TBILI": VarSpec("labevents", [r"bilirubin,? total"],     "mg/dL", 0.0,   40.0, None),
    "INR":   VarSpec("labevents", [r"^inr$"],                 None,    0.5,   20.0, None),
}

AGG_FUNCS = ["mean", "median", "min", "max", "std"]
WINDOW_HOURS_DEFAULT = 24

# ---------------------------
# Helpers
# ---------------------------

def f_to_c(x: pd.Series) -> pd.Series:
    mask = x > 45  # if it looks like Fahrenheit, convert
    y = x.copy()
    y.loc[mask] = (x.loc[mask] - 32.0) * (5.0 / 9.0)
    return y

CONVERTERS = {"f_to_c": f_to_c}

def read_csv_case_insensitive(datadir: str, name: str) -> pd.DataFrame:
    files = os.listdir(datadir)
    candidates = [f for f in files if f.lower() == name.lower()]
    if not candidates:
        logging.warning("Missing %s in %s — returning empty DataFrame.", name, datadir)
        return pd.DataFrame()
    df = pd.read_csv(os.path.join(datadir, candidates[0]), low_memory=False)
    return df

def to_parquet_safe(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        logging.warning("Parquet save failed for %s (%s). Install 'pyarrow' or 'fastparquet' to enable.", path, e)

def norm_upper(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=str.upper) if not df.empty else df

# ---------------------------
# Cohort
# ---------------------------

def make_cohort(admissions: pd.DataFrame, icustays: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    if admissions.empty or icustays.empty:
        return pd.DataFrame()

    admissions = norm_upper(admissions)
    icustays   = norm_upper(icustays)

    # Parse times
    for col in ["INTIME", "OUTTIME"]:
        if col in icustays.columns:
            icustays[col] = pd.to_datetime(icustays[col], errors="coerce")
    for col in ["ADMITTIME", "DISCHTIME"]:
        if col in admissions.columns:
            admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

    # First ICU stay per hospital admission (deterministic)
    if "HADM_ID" in icustays.columns:
        icu_sorted = icustays.sort_values(["HADM_ID", "INTIME"]).drop_duplicates("HADM_ID", keep="first")
        if {"HADM_ID", "SUBJECT_ID"}.issubset(admissions.columns):
            cohort = icu_sorted.merge(admissions, on=["HADM_ID", "SUBJECT_ID"], how="left", suffixes=("_ICU", "_ADM"))
        else:
            cohort = icu_sorted.copy()
    else:
        if "SUBJECT_ID" in icustays.columns:
            icu_sorted = icustays.sort_values(["SUBJECT_ID", "INTIME"]).drop_duplicates("SUBJECT_ID", keep="first")
        else:
            icu_sorted = icustays.drop_duplicates(subset=["ICUSTAY_ID"])
        cohort = icu_sorted.copy()

    keep = [c for c in ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "ADMITTIME", "DISCHTIME"] if c in cohort.columns]
    cohort = cohort[keep].rename(columns=str.lower)

    if "intime" not in cohort.columns:
        raise RuntimeError("ICUSTAYS does not contain INTIME; cannot define ICU window.")

    cohort["window_start"] = cohort["intime"]
    cohort["window_stop"]  = cohort["intime"] + pd.to_timedelta(window_hours, unit="h")
    cohort = cohort.dropna(subset=["window_start"])
    return cohort

# ---------------------------
# Selection & Extraction
# ---------------------------

def select_items_by_label(d_ref: pd.DataFrame, varspec: VarSpec) -> pd.DataFrame:
    if d_ref.empty:
        return d_ref
    d_ref = norm_upper(d_ref)
    label_col = "LABEL" if "LABEL" in d_ref.columns else ("LABEL_X" if "LABEL_X" in d_ref.columns else None)
    if not label_col:
        return pd.DataFrame()
    regex = re.compile("|".join(varspec.patterns), flags=re.IGNORECASE)
    return d_ref[d_ref[label_col].astype(str).str.contains(regex, na=False)]

def extract_timeseries_chartevents(chartevents: pd.DataFrame, d_items: pd.DataFrame,
                                   cohort: pd.DataFrame, varname: str, spec: VarSpec,
                                   window_hours: int) -> pd.DataFrame:
    if chartevents.empty or d_items.empty or cohort.empty:
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    chartevents = norm_upper(chartevents)
    d_items     = norm_upper(d_items)

    required = {"ITEMID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM"}
    missing  = [c for c in required if c not in chartevents.columns]
    if missing:
        logging.warning("CHARTEVENTS missing %s — skipping %s.", missing, varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    cols_ref = [c for c in ["ITEMID", "LABEL", "UNITNAME"] if c in d_items.columns]
    ce = chartevents.merge(d_items[cols_ref], on="ITEMID", how="left")

    matched = select_items_by_label(d_items, spec)
    if "ITEMID" not in matched.columns or matched.empty:
        logging.warning("D_ITEMS matched no ITEMIDs for %s.", varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    ce = ce[ce["ITEMID"].isin(matched["ITEMID"])].copy()

    ce["CHARTTIME"] = pd.to_datetime(ce["CHARTTIME"], errors="coerce")
    ce = ce.rename(columns={"CHARTTIME": "charttime", "ICUSTAY_ID": "icustay_id", "VALUENUM": varname})

    ce = ce[pd.to_numeric(ce[varname], errors="coerce").notna()]
    ce[varname] = ce[varname].astype(float)

    m = ce.merge(cohort[["icustay_id", "window_start", "window_stop"]], on="icustay_id", how="inner")
    m = m[(m["charttime"] >= m["window_start"]) & (m["charttime"] <= m["window_stop"])].copy()

    if spec.convert in CONVERTERS:
        m[varname] = CONVERTERS[spec.convert](m[varname])

    if spec.physiologic_min is not None:
        m[varname] = m[varname].clip(lower=spec.physiologic_min)
    if spec.physiologic_max is not None:
        m[varname] = m[varname].clip(upper=spec.physiologic_max)

    return m[["icustay_id", "charttime", varname]]

def extract_timeseries_labevents(labevents: pd.DataFrame, d_labitems: pd.DataFrame,
                                 cohort: pd.DataFrame, varname: str, spec: VarSpec,
                                 window_hours: int) -> pd.DataFrame:
    if labevents.empty or d_labitems.empty or cohort.empty:
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    labevents  = norm_upper(labevents)
    d_labitems = norm_upper(d_labitems)

    if "ITEMID" not in labevents.columns:
        logging.warning("LABEVENTS missing ITEMID — skipping %s.", varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    cols_ref = [c for c in ["ITEMID", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE", "UNITNAME"] if c in d_labitems.columns]
    le = labevents.merge(d_labitems[cols_ref], on="ITEMID", how="left")

    matched = select_items_by_label(d_labitems, spec)
    if "ITEMID" not in matched.columns or matched.empty:
        logging.warning("D_LABITEMS matched no ITEMIDs for %s.", varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    le = le[le["ITEMID"].isin(matched["ITEMID"])].copy()

    if "CHARTTIME" not in le.columns:
        logging.warning("LABEVENTS missing CHARTTIME — skipping %s.", varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])
    le["CHARTTIME"] = pd.to_datetime(le["CHARTTIME"], errors="coerce")
    le = le.rename(columns={"CHARTTIME": "charttime", "VALUENUM": varname})

    # Prefer HADM linkage; fallback ICUSTAY_ID
    if "HADM_ID" in le.columns and "hadm_id" in cohort.columns:
        le = le.merge(cohort[["hadm_id", "icustay_id", "window_start", "window_stop"]],
                      left_on="HADM_ID", right_on="hadm_id", how="inner")
    elif "ICUSTAY_ID" in le.columns and "icustay_id" in cohort.columns:
        le = le.merge(cohort[["icustay_id", "window_start", "window_stop"]],
                      on="icustay_id", how="inner")
    else:
        logging.warning("Cannot link LABEVENTS to cohort for %s (need HADM_ID or ICUSTAY_ID).", varname)
        return pd.DataFrame(columns=["icustay_id", "charttime", varname])

    le = le[pd.to_numeric(le[varname], errors="coerce").notna()]
    le[varname] = le[varname].astype(float)

    le = le[(le["charttime"] >= le["window_start"]) & (le["charttime"] <= le["window_stop"])].copy()

    if spec.physiologic_min is not None:
        le[varname] = le[varname].clip(lower=spec.physiologic_min)
    if spec.physiologic_max is not None:
        le[varname] = le[varname].clip(upper=spec.physiologic_max)

    return le[["icustay_id", "charttime", varname]]

# ---------------------------
# Aggregation
# ---------------------------

def hourly_aggregate(ts: pd.DataFrame, varname: str) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame(columns=["icustay_id",
                                     f"{varname}_mean_mean", f"{varname}_mean_min", f"{varname}_mean_max", f"{varname}_mean_std",
                                     f"{varname}_median_mean", f"{varname}_min_min", f"{varname}_max_max", f"{varname}_std_mean",
                                     f"{varname}_hours_fraction"])
    ts = ts.copy()
    ts["hour"] = ts["charttime"].dt.floor("H")
    agged = ts.groupby(["icustay_id", "hour"])[varname].agg(AGG_FUNCS).reset_index()

    global_stats = agged.groupby("icustay_id")[AGG_FUNCS].agg({
        "mean":   ["mean", "min", "max", "std"],
        "median": ["mean"],
        "min":    ["min"],
        "max":    ["max"],
        "std":    ["mean"],
    })
    global_stats.columns = [f"{varname}_{a}_{b}" for a, b in global_stats.columns]
    global_stats = global_stats.reset_index()

    coverage = agged.groupby("icustay_id").size().rename(f"{varname}_hours_present").reset_index()
    out = global_stats.merge(coverage, on="icustay_id", how="left")
    out[f"{varname}_hours_fraction"] = out[f"{varname}_hours_present"].clip(0, 24) / 24.0
    out = out.drop(columns=[f"{varname}_hours_present"], errors="ignore")
    return out

# ---------------------------
# Build Features
# ---------------------------

def build_feature_matrix(datadir: str, outdir: str, window_hours: int) -> Tuple[pd.DataFrame, Dict]:
    os.makedirs(outdir, exist_ok=True)
    logging.info("Loading core tables from %s", datadir)

    admissions  = read_csv_case_insensitive(datadir, "ADMISSIONS.csv")
    icustays    = read_csv_case_insensitive(datadir, "ICUSTAYS.csv")
    chartevents = read_csv_case_insensitive(datadir, "CHARTEVENTS.csv")
    d_items     = read_csv_case_insensitive(datadir, "D_ITEMS.csv")
    labevents   = read_csv_case_insensitive(datadir, "LABEVENTS.csv")
    d_labitems  = read_csv_case_insensitive(datadir, "D_LABITEMS.csv")

    cohort = make_cohort(admissions, icustays, window_hours)
    if cohort.empty:
        raise RuntimeError("Cohort is empty — check that ADMISSIONS.csv and ICUSTAYS.csv are present and non-empty.")

    # Save cohort
    to_parquet_safe(cohort, os.path.join(outdir, "cohort.parquet"))
    cohort.to_csv(os.path.join(outdir, "cohort.csv"), index=False)

    # Extract & aggregate
    features = cohort[["icustay_id"]].drop_duplicates().set_index("icustay_id")
    feature_meta: Dict[str, dict] = {}

    for varname, spec in VARS.items():
        logging.info("Extracting %s from %s", varname, spec.source)
        if spec.source == "chartevents":
            ts = extract_timeseries_chartevents(chartevents, d_items, cohort, varname, spec, window_hours)
        else:
            ts = extract_timeseries_labevents(labevents, d_labitems, cohort, varname, spec, window_hours)

        agged = hourly_aggregate(ts, varname)
        if not agged.empty:
            agged = agged.set_index("icustay_id")
            features = features.join(agged, how="left")
            feature_meta[varname] = asdict(spec)
        else:
            logging.warning("No data matched for %s", varname)

    # If nothing matched, bail out gracefully
    if features.shape[1] == 0:
        logging.error("No features extracted — none of the VARS matched available events in the demo.")
        raw = features.reset_index()
        to_parquet_safe(raw, os.path.join(outdir, "features_raw.parquet"))
        raw.to_csv(os.path.join(outdir, "features_raw.csv"), index=False)
        with open(os.path.join(outdir, "feature_metadata.json"), "w") as f:
            json.dump({"window_hours": window_hours, "variables": {}, "notes": ["Empty features"]}, f, indent=2)
        return features.reset_index(), {}

    # Save raw features
    raw = features.reset_index()
    to_parquet_safe(raw, os.path.join(outdir, "features_raw.parquet"))
    raw.to_csv(os.path.join(outdir, "features_raw.csv"), index=False)

    # Impute + scale
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_imputed = imputer.fit_transform(features)
    imputed_cols = list(features.columns) + [f"_missing{i}" for i in range(X_imputed.shape[1] - features.shape[1])]
    X_imp_df = pd.DataFrame(X_imputed, columns=imputed_cols, index=features.index)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=imputed_cols, index=features.index)

    clean = X_scaled_df.reset_index()
    to_parquet_safe(clean, os.path.join(outdir, "features_clean.parquet"))
    clean.to_csv(os.path.join(outdir, "features_clean.csv"), index=False)

    dump(imputer, os.path.join(outdir, "imputer_median.joblib"))
    dump(scaler,   os.path.join(outdir, "scaler_robust.joblib"))

    meta = {
        "window_hours": window_hours,
        "variables": feature_meta,
        "agg_funcs": AGG_FUNCS,
        "scaler": "RobustScaler(median/IQR)",
        "imputer": "SimpleImputer(median) + missingness indicators",
        "notes": [
            "Dictionary tables matched by regex; events filtered to first window.",
            "Temperature: auto °F→°C for plausible values.",
            "Values clipped to physiologic ranges.",
            "Parquet saves best-effort; CSV always saved.",
        ],
    }
    with open(os.path.join(outdir, "feature_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logging.info("Built feature matrix with shape %s", X_scaled_df.shape)
    return X_scaled_df, meta

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III demo preprocessing to features")
    parser.add_argument("--datadir", required=True, help="Directory with MIMIC-III demo CSVs")
    parser.add_argument("--outdir", required=True, help="Output directory for features")
    parser.add_argument("--icu-window-hours", type=int, default=WINDOW_HOURS_DEFAULT,
                        help="Window from ICU intime (hours)")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    _X, _meta = build_feature_matrix(args.datadir, args.outdir, args.icu_window_hours)
    logging.info("Done. Outputs written to %s", args.outdir)

if __name__ == "__main__":
    main()
