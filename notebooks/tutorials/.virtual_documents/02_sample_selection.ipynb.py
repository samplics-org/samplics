#get_ipython().run_line_magic("load_ext", " lab_black ")

# In this cell, all the necessary python packages and classes are imported in the workspace.
import numpy as np
import pandas as pd

# first run (from the terminal): pip3 install samplics
from samplics import Sample


psu_frame = pd.read_csv("psu_frame.csv")

psu_frame.head(25)


psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}
print(psu_sample_size)


from samplics import array_to_dict

frame_size = array_to_dict(psu_frame["region"])
print(f"The number of clusters per stratum is: {frame_size} \n")

psu_sample_size = frame_size.copy()
psu_sample_size["East"] = 3
psu_sample_size["North"] = 2
psu_sample_size["South"] = 3
psu_sample_size["West"] = 2
print(f"The sample size per stratum is: {psu_sample_size}")


stage1_design = Sample(method="pps-sys", stratification=True, with_replacement=False)

psu_frame["psu_prob"] = stage1_design.inclusion_probs(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"],
    psu_frame["number_households_census"],
    )

psu_frame.head(15)


np.random.seed(23)

psu_frame["psu_sample"], psu_frame["psu_hits"], psu_frame["psu_probs"] = stage1_design.select(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"], 
    psu_frame["number_households_census"]
    )

psu_frame.head(15)


np.random.seed(23)

psu_sample = stage1_design.select(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"], 
    psu_frame["number_households_census"],
    sample_only = True
    )

psu_sample


np.random.seed(23)

stage1_sampford = Sample(method="pps-rs", stratification=True, with_replacement=False)

psu_sample_sampford = stage1_sampford.select(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"], 
    psu_frame["number_households_census"],
    sample_only=True
    )

psu_sample_sampford.head(15)


# Create a synthetic second stage frame
census_size = psu_frame.loc[psu_frame["psu_sample"]==1, "number_households_census"].values
stratum_names = psu_frame.loc[psu_frame["psu_sample"]==1, "region"].values
cluster = psu_frame.loc[psu_frame["psu_sample"]==1, "cluster"].values

np.random.seed(15)

listing_size = np.zeros(census_size.size)
for k in range(census_size.size):
    listing_size[k] = np.random.normal(1.05*census_size[k], 0.15*census_size[k])
    
listing_size = listing_size.astype(int)
hh_id = rr_id = cl_id = []
for k, s in enumerate(listing_size):
    hh_k1 = np.char.array(np.repeat(stratum_names[k], s)).astype(str)
    hh_k2 = np.char.array(np.arange(1, s+1)).astype(str)
    cl_k = np.repeat(cluster[k], s)
    hh_k = np.char.add(np.char.array(cl_k).astype(str), hh_k2)
    hh_id = np.append(hh_id, hh_k)
    rr_id = np.append(rr_id, hh_k1)
    cl_id = np.append(cl_id, cl_k)

ssu_frame = pd.DataFrame(cl_id.astype(int))
ssu_frame.rename(columns={0: "cluster"}, inplace=True)
ssu_frame["region"] = rr_id
ssu_frame["household"] = hh_id

ssu_frame.head(15)


psu_sample = psu_frame.loc[psu_frame["psu_sample"]==1]
ssu_counts = ssu_frame.groupby("cluster").count()
ssu_counts.drop(columns="region", inplace=True)
ssu_counts.reset_index(inplace=True)
ssu_counts.rename(columns={"household":"number_households_listed"}, inplace=True)

pd.merge(
    psu_sample[["cluster", "region", "number_households_census"]], 
    ssu_counts[["cluster", "number_households_listed"]], on=["cluster"]
    )


stage2_design = Sample(method="srs", stratification=True, with_replacement=False)

ssu_frame["ssu_prob"] = stage2_design.inclusion_probs(
    ssu_frame["household"], 15, ssu_frame["cluster"]
    )

ssu_frame.sample(20)


np.random.seed(11)
ssu_sample, ssu_hits, ssu_probs = stage2_design.select(ssu_frame["household"], 15, ssu_frame["cluster"])

ssu_frame["ssu_sample"] = ssu_sample
ssu_frame["ssu_hits"] = ssu_hits
ssu_frame["ssu_probs"] = ssu_probs

ssu_frame[ssu_frame["ssu_sample"]==1].sample(15)


rates = np.repeat(15, 10) / ssu_counts["number_households_listed"].values

ssu_rates = dict(zip(np.unique(ssu_frame["cluster"]), rates))

ssu_rates


np.random.seed(22)

stage2_design2 = Sample(method="sys", stratification=True, with_replacement=False)

ssu_sample_r, ssu_hits_r, _ = stage2_design2.select(
    ssu_frame["household"], stratum=ssu_frame["cluster"], samp_rate=ssu_rates
    )

ssu_sample2 = pd.DataFrame(
    data={
        "household":ssu_frame["household"], 
        "ssu_sample_r":ssu_sample_r,
        "ssu_hits_r":ssu_hits_r
    })

ssu_sample2.head(25)


psu_sample[["cluster", "region", "psu_prob"]].to_csv("psu_sample.csv")

ssu_sample = ssu_frame.loc[ssu_frame["ssu_sample"]==1]
ssu_sample[["cluster", "household", "ssu_prob"]].to_csv("ssu_sample.csv")
