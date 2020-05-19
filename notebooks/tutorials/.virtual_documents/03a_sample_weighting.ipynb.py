import numpy as np
import pandas as pd

import samplics
from samplics.weighting import SampleWeight


psu_sample = pd.read_csv("psu_sample.csv")
ssu_sample = pd.read_csv("ssu_sample.csv")

full_sample = pd.merge(
    psu_sample[["cluster", "region", "psu_prob"]], 
    ssu_sample[["cluster", "household", "ssu_prob"]], 
    on="cluster")

full_sample["inclusion_prob"] = full_sample["psu_prob"] * full_sample["ssu_prob"] 
full_sample["design_weight"] = 1 / full_sample["inclusion_prob"] 

full_sample.head(15)


np.random.seed(7)
full_sample["response_status"] = np.random.choice(
    ["ineligible","respondent", "non-respondent","unknown"], 
    size=full_sample.shape[0], 
    p=(0.10, 0.70, 0.15, 0.05)
    )

full_sample[["cluster", "region","design_weight", "response_status"]].head(15)


status_mapping = {
    "in": "ineligible", "rr": "respondent", "nr": "non-respondent", "uk":"unknown"
    }

full_sample["nr_weight"] = SampleWeight().adjust(
    samp_weight=full_sample["design_weight"], 
    adjust_class=full_sample[["region", "cluster"]], 
    resp_status=full_sample["response_status"], 
    resp_dict=status_mapping
    )

full_sample["nr_weight2"] = SampleWeight().adjust(
    samp_weight=full_sample["design_weight"], 
    adjust_class=full_sample[["region", "cluster"]], 
    resp_status=full_sample["response_status"], 
    resp_dict=status_mapping,
    unknown_to_inelig = False
    )

full_sample[["cluster", "region","design_weight", "response_status", "nr_weight", "nr_weight2"]].drop_duplicates().head(15)


response_status2 = np.repeat(100, full_sample["response_status"].shape[0])
response_status2[full_sample["response_status"]=="non-respondent"] = 200
response_status2[full_sample["response_status"]=="respondent"] = 300
response_status2[full_sample["response_status"]=="unknown"] = 999

pd.crosstab(response_status2, full_sample["response_status"])


status_mapping2 = {"in": 100, "nr": 200, "rr": 300, "uk": 999}

full_sample["nr_weight3"] = SampleWeight().adjust(
    samp_weight=full_sample["design_weight"], 
    adjust_class=full_sample[["region", "cluster"]], 
    resp_status=response_status2, 
    resp_dict=status_mapping2
    )

full_sample[["cluster", "region", "response_status", "nr_weight", "nr_weight3"]].drop_duplicates().head(15)


response_status3 = np.repeat("in", full_sample["response_status"].shape[0])
response_status3[full_sample["response_status"]=="non-respondent"] = "nr"
response_status3[full_sample["response_status"]=="respondent"] = "rr"
response_status3[full_sample["response_status"]=="unknown"] = "uk"

full_sample["nr_weight4"] = SampleWeight().adjust(
    samp_weight=full_sample["design_weight"], 
    adjust_class=full_sample[["region", "cluster"]], 
    resp_status=response_status3
    )

full_sample[["cluster", "region", "response_status", "nr_weight", "nr_weight4"]].drop_duplicates().head(15)


# Just dropping a couple of variables not needed for the rest of the tutorial
full_sample.drop(
    columns=["psu_prob", "ssu_prob", "inclusion_prob", "nr_weight2", "nr_weight3", "nr_weight4"], 
    inplace=True
)


census_households = {"East":3700, "North": 1500, "South": 2800, "West":6500}

full_sample["ps_weight"] = SampleWeight().poststratify(
    samp_weight=full_sample["nr_weight"], control=census_households, domain=full_sample["region"]
    )

full_sample.head(15)


sum_of_weights = full_sample[["region", "nr_weight", "ps_weight"]].groupby("region").sum()
sum_of_weights.reset_index(inplace=True)
sum_of_weights.head()


full_sample["ps_adjust_fct"] = round(full_sample["ps_weight"] / full_sample["nr_weight"], 12)

pd.crosstab(full_sample["ps_adjust_fct"] , full_sample["region"])


known_ratios = {"East":0.25, "North": 0.10, "South": 0.20, "West":0.45}
full_sample["ps_weight2"] = SampleWeight().poststratify(
    samp_weight=full_sample["nr_weight"], factor=known_ratios, domain=full_sample["region"]
    )

full_sample.head()


sum_of_weights2 = full_sample[["region", "nr_weight", "ps_weight2"]].groupby("region").sum()
sum_of_weights2.reset_index(inplace=True)
sum_of_weights2["ratio"] = sum_of_weights2["ps_weight2"] / sum(sum_of_weights2["ps_weight2"])
sum_of_weights2.head()


np.random.seed(150)
full_sample["education"] = np.random.choice(("Low", "Medium", "High"), size=150, p=(0.40, 0.50, 0.10))
full_sample["poverty"] = np.random.choice((0, 1), size=150, p=(0.70, 0.30))
full_sample["under_five"] = np.random.choice((0,1,2,3,4,5), size=150, p=(0.05, 0.35, 0.25, 0.20, 0.10, 0.05))
full_sample.head()


totals = {"poverty": 4700, "under_five": 30800}

full_sample["calib_weight"] = SampleWeight().calibrate(
    full_sample["nr_weight"], full_sample[["poverty", "under_five"]], totals
    )

full_sample[["cluster", "region", "household", "nr_weight", "calib_weight"]].head(15)


poverty = full_sample["poverty"]
under_5 = full_sample["under_five"]
nr_weight = full_sample["nr_weight"]
calib_weight = full_sample["calib_weight"]

print(f"Total estimated number of poor households was {sum(poverty*nr_weight):.2f} before and {sum(poverty*calib_weight):.2f} after adjustment \n")
print(f"Total estimated number of children under 5 was {sum(under_5*nr_weight):.2f} before and {sum(under_5*calib_weight):.2f} after adjustment \n")


totals_by_domain = {
    "East": {"poverty": 1200, "under_five": 6300}, 
    "North": {"poverty": 200, "under_five": 4000}, 
    "South": {"poverty": 1100, "under_five": 6500}, 
    "West": {"poverty": 2200, "under_five": 14000}
    }

full_sample["calib_weight_d"] = SampleWeight().calibrate(
    full_sample["nr_weight"], full_sample[["poverty", "under_five"]], totals_by_domain, full_sample["region"]
    )

full_sample[["cluster", "region", "household", "nr_weight", "calib_weight", "calib_weight_d"]].head(15)


print(f"The number of households using the overall GREG is: {sum(full_sample['calib_weight']):.2f} \n")
print(f"The number of households using the domain GREG is: {sum(full_sample['calib_weight_d']):.2f} \n")


totals_by_domain = {
    "East": {"poverty": 1200, "under_five": 6300}, 
    "North": {"poverty": 200, "under_five": 4000}, 
    "South": {"poverty": 1100, "under_five": 6500}, 
    "West": {"poverty": 2200, "under_five": 14000}
    }

calib_weight3 = SampleWeight().calibrate(
    full_sample["nr_weight"], 
    full_sample[["poverty", "under_five"]], 
    totals_by_domain, 
    full_sample["region"], 
    additive=True
    )

under_5 = np.array(full_sample["under_five"])
print(f"Each column can be used to estimate a domain: {np.sum(np.transpose(calib_weight3) * under_5, axis=1)}\n")
print(f"The number of households using the overall GREG is: {sum(full_sample['calib_weight']):.2f} \n")
print(f"The number of households using the domain GREG is: {sum(full_sample['calib_weight_d']):.2f} - with ADDITIVE=FALSE\n")
print(f"The number of households using the domain GREG is: {np.sum(np.transpose(calib_weight3)):.2f} - with ADDITIVE=TRUE \n")


full_sample["norm_weight"] = SampleWeight().normalize(full_sample["nr_weight"])


full_sample[["cluster", "region", "nr_weight", "norm_weight"]].head(25)

print((full_sample.shape[0], full_sample["norm_weight"].sum()))


full_sample["norm_weight2"] = SampleWeight().normalize(full_sample["nr_weight"], control=300)

print(full_sample["norm_weight2"].sum())


full_sample["norm_weight3"] = SampleWeight().normalize(full_sample["nr_weight"], domain=full_sample["region"])

weight_sum = full_sample.groupby(["region"]).sum()
weight_sum.reset_index(inplace=True)
weight_sum[["region", "nr_weight", "norm_weight", "norm_weight3"]]


norm_level = {"East": 10, "North": 20, "South": 30, "West": 50}

full_sample["norm_weight4"] = SampleWeight().normalize(full_sample["nr_weight"], norm_level, full_sample["region"])

weight_sum = full_sample.groupby(["region"]).sum()
weight_sum.reset_index(inplace=True)
weight_sum[["region", "nr_weight", "norm_weight", "norm_weight2", "norm_weight3", "norm_weight4",]]
