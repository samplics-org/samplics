from IPython.core.display import Image, display

import numpy as np
import pandas as pd

import samplics 
from samplics.estimation import TaylorEstimator, ReplicateEstimator


nhanes2f = pd.read_csv("../../../datasets/docs/nhanes2f.csv")

nhanes2f[["psuid", "stratid", "highbp", "highlead", "finalwgt"]].head()


Image(filename="zinc_mean_stata_str.png")


zinc_mean_str = TaylorEstimator("mean").estimate(
    y=nhanes2f["zinc"],
    samp_weight=nhanes2f["finalwgt"],
    stratum=nhanes2f["stratid"],
    psu=nhanes2f["psuid"],
    remove_nan=True,
)

print(zinc_mean_str)


Image(filename="zinc_mean_stata_nostr.png")


zinc_mean_nostr = TaylorEstimator("mean").estimate(
    y=nhanes2f["zinc"], samp_weight=nhanes2f["finalwgt"], psu=nhanes2f["psuid"], remove_nan=True
)

print(zinc_mean_nostr)


Image(filename="ratio_highbp_highlead.png")


ratio_bp_lead = TaylorEstimator("ratio").estimate(
    y=nhanes2f["highbp"],
    samp_weight=nhanes2f["finalwgt"],
    x=nhanes2f["highlead"],
    stratum=nhanes2f["stratid"],
    psu=nhanes2f["psuid"],
    remove_nan=True,
)

print(ratio_bp_lead)


nmihs_bs = pd.read_csv("../../../datasets/docs/nmihs_bs.csv")

nmihs_bs.describe()


Image(filename="mean_birthwgt_bs.png")


# rep_wgt_boot = nmihsboot.loc[:, "bsrw1":"bsrw1000"]

birthwgt = ReplicateEstimator("bootstrap", "mean").estimate(
    y=nmihs_bs["birthwgt"],
    samp_weight=nmihs_bs["finwgt"],
    rep_weights=nmihs_bs.loc[:, "bsrw1":"bsrw1000"],
    remove_nan=True,
)

print(birthwgt)


nhanes2brr = pd.read_csv("../../../datasets/docs/nhanes2brr.csv")

nhanes2brr.describe()


Image(filename="ratio_weight_height_brr.png")


brr = ReplicateEstimator("brr", "ratio")

ratio_wgt_hgt = brr.estimate(
    y=nhanes2brr["weight"],
    samp_weight=nhanes2brr["finalwgt"],
    x=nhanes2brr["height"],
    rep_weights=nhanes2brr.loc[:, "brr_1":"brr_32"],
    remove_nan=True,
)

print(ratio_wgt_hgt)


nhanes2jknife = pd.read_csv("../../../datasets/docs/nhanes2jknife.csv")

nhanes2jknife.describe()


Image(filename="ratio_weight_height_jknife.png")


jackknife = ReplicateEstimator("jackknife", "ratio")

ratio_wgt_hgt2 = jackknife.estimate(
    y=nhanes2jknife["weight"],
    samp_weight=nhanes2jknife["finalwgt"],
    x=nhanes2jknife["height"],
    rep_weights=nhanes2jknife.loc[:, "jkw_1":"jkw_62"],
    rep_coefs=0.5,
    remove_nan=True,
)

print(ratio_wgt_hgt2)
