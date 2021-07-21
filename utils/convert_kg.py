# %%
from os import path

import pandas as pd

# %% Convert to chinese with commas

input_path = "/Users/alexandreduperre/Documents/SEAR-827/medical_kg_raw_chinese.csv"
kg = pd.read_csv(input_path, names=["0", "1", "2"])

print(kg)
#%%
output_path = "/Users/alexandreduperre/Documents/SEAR-827/medical_kg_chinese_commas.txt"
with open(output_path, "w+") as f:
    for i in range(len(kg)):
        f.write(kg.loc[i, "0"] + "," + kg.loc[i, "1"] + "," + kg.loc[i, "2"] + "\n")

    f.close()

# %% Convert to format with tabs

input_path = "/Users/alexandreduperre/Documents/SEAR-827/medical_kg_raw_en.csv"
kg = pd.read_csv(input_path, names=["0", "1", "2", "3", "4", "5"])

print(kg)
kg.iloc[23:26]

# %%
output_path = "/Users/alexandreduperre/Documents/SEAR-827/medical_kg_en.spo"
with open(output_path, "w+") as f:
    for i in range(len(kg)):
        if str(kg.loc[i, "3"]) == "nan":
            try:
                f.write(
                    kg.loc[i, "0"].strip().lower()
                    + "\t"
                    + kg.loc[i, "1"].strip()
                    + "\t"
                    + kg.loc[i, "2"].strip().lower()
                    + "\n"
                )

            except:
                pass

f.close()
# %%
