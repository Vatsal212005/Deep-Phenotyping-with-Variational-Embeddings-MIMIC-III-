import pandas as pd
import os

datadir = r"C:\Users\ironm\OneDrive\Desktop\Resume Project\Dataset\mimic-iii-clinical-database-demo-1.4"

# Load just the necessary columns
chartevents = pd.read_csv(os.path.join(datadir, "CHARTEVENTS.csv"), usecols=["itemid", "valuenum"])
chartevents.columns = [c.lower() for c in chartevents.columns]

print("CHARTEVENTS rows:", len(chartevents))

# Example important ITEMIDs we care about (from D_ITEMS check)
important_itemids = [211, 220045, 220210, 646, 677]  # HR, RR, Temp, SpO2, etc.

for iid in important_itemids:
    subset = chartevents[chartevents["itemid"] == iid]
    if not subset.empty:
        n_values = subset["valuenum"].notna().sum()
        print(f"✅ ITEMID {iid} found with {n_values} numeric values")
    else:
        print(f"⚠️ ITEMID {iid} not found in CHARTEVENTS")
