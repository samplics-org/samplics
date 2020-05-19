import numpy as np
import pandas as pd

import samplics
from samplics.weighting import ReplicateWeight


psu_sample = pd.read_csv("psu_sample.csv")
ssu_sample = pd.read_csv("ssu_sample.csv")

full_sample = pd.merge(
    psu_sample[["cluster", "region", "psu_prob"]], 
    ssu_sample[["cluster", "household", "ssu_prob"]], 
    on="cluster")

full_sample["inclusion_prob"] = full_sample["psu_prob"] * full_sample["ssu_prob"] 
full_sample["design_weight"] = 1 / full_sample["inclusion_prob"] 

full_sample.head(15)


import scipy
scipy.linalg.hadamard(8)


brr = ReplicateWeight(method="brr", stratification=False)
brr_wgt = brr.replicate(full_sample["design_weight"], full_sample["cluster"])

brr_wgt.drop_duplicates().head(10)


fay = ReplicateWeight(method="brr", stratification=False, fay_coef=0.3)
fay_wgt = fay.replicate(
    full_sample["design_weight"], 
    full_sample["cluster"], 
    rep_prefix="fay_weight_",
    psu_varname="cluster", 
    str_varname="stratum"
)

fay_wgt.drop_duplicates().head(10)


bootstrap = ReplicateWeight(method="bootstrap", stratification=False, number_reps=50)
boot_wgt = bootstrap.replicate(full_sample["design_weight"], full_sample["cluster"])

boot_wgt.drop_duplicates().head(10)


jackknife = ReplicateWeight(method="jackknife", stratification=False)
jkn_wgt = jackknife.replicate(full_sample["design_weight"], full_sample["cluster"])

jkn_wgt.drop_duplicates().head(10)


jackknife = ReplicateWeight(method="jackknife", stratification=True)
jkn_wgt = jackknife.replicate(full_sample["design_weight"], full_sample["cluster"], full_sample["region"])

jkn_wgt.drop_duplicates().head(10)


#jackknife = ReplicateWeight(method="jackknife", stratification=True)
jkn_wgt = jackknife.replicate(
    full_sample["design_weight"], full_sample["cluster"], full_sample["region"], rep_coefs=True
)

jkn_wgt.drop_duplicates().sort_values(by="_stratum").head(15)


#fay = ReplicateWeight(method="brr", stratification=False, fay_coef=0.3)
fay_wgt = fay.replicate(
    full_sample["design_weight"], 
    full_sample["cluster"], 
    rep_prefix="fay_weight_",
    psu_varname="cluster", 
    str_varname="stratum",
    rep_coefs=True
)

fay_wgt.drop_duplicates().head(10)
