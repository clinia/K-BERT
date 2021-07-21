#%%
import pandas as pd

#%%

paths = {
    "brand": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/brand",
    "condition": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/condition",
    "establishment": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/establishment",
    "product": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/product",
    "profession": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/profession",
    "service": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/service",
    "specialty": "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/specialty",
}
data_sources = []
for source, path in paths.items():
    data = pd.read_csv(path + "/name.csv", names=["entity"], engine="python")
    data["relation"] = source
    data_sources.append(data)

#%%
data_sources[2]

#%%
# Create a first level kg and export
kg = pd.concat(data_sources, ignore_index=True).dropna().reset_index(drop=True)
# kg.to_csv(".clinia_kg.csv")

#%%
output_path = "/Users/alexandreduperre/Documents/SEAR-827/Clinia kg/clinia_kg.spo"
with open(output_path, "w+") as f:
    for i in range(len(kg)):
        # print(kg.loc[i])
        f.write(
            kg.loc[i, "entity"].strip().lower() + "\t" + "type of" + "\t" + kg.loc[i, "relation"].strip().lower() + "\n"
        )

f.close()

# %%
