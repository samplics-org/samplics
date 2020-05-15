import numpy as np
import pandas as pd
get_ipython().getoutput("pip3 install --upgrade samplics")
import samplics
from samplics.sampling import Sample


psu_frame = pd.read_csv("../../../datasets/docs/psu_frame.csv")

psu_frame.head(25)


psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}

print(f"\nThe sample size per domain is: {psu_sample_size}\n")


from samplics import array_to_dict

frame_size = array_to_dict(psu_frame["region"])
print(f"\nThe number of clusters per stratum is: {frame_size}")

psu_sample_size = frame_size.copy()
psu_sample_size["East"] = 3
psu_sample_size["North"] = 2
psu_sample_size["South"] = 3
psu_sample_size["West"] = 2
print(f"\nThe sample size per stratum is: {psu_sample_size}\n")


stage1_design = Sample(method="pps-sys", stratification=True, with_replacement=False)

psu_frame["psu_prob"] = stage1_design.inclusion_probs(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"],
    psu_frame["number_households_census"],
    )

nb_obs = 15
print(f"\nFirst {nb_obs} observations of the PSU frame \n")
psu_frame.head(nb_obs)


np.random.seed(23)

psu_frame["psu_sample"], psu_frame["psu_hits"], psu_frame["psu_probs"] = stage1_design.select(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"], 
    psu_frame["number_households_census"]
    )

nb_obs = 15
print(f"\nFirst {nb_obs} observations of the PSU frame with the sampling information \n")
psu_frame.head(nb_obs)


np.random.seed(23)

psu_sample = stage1_design.select(
    psu_frame["cluster"], 
    psu_sample_size, 
    psu_frame["region"], 
    psu_frame["number_households_census"],
    sample_only = True
    )

print("\nPSU sample without the non-sampled units\n")
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






