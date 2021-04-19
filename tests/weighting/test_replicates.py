import pandas as pd

from samplics.weighting import ReplicateWeight


test_data = pd.read_csv("./tests/weighting/small_data_for_testing_replicate_weights.csv")

cluster_id = test_data["psu"].astype(str)
stratum_id = test_data["stratum"].astype(str)
sample_wgt = test_data["weight"]

nb_reps = 5

"""Balanced Repeated Replicates (BRR) WITHOUT stratification"""
no_str_brr = ReplicateWeight(
    method="brr", stratification=False, number_reps=nb_reps, fay_coef=0.05
)
no_str_brr_wgt = no_str_brr.replicate(sample_wgt, cluster_id)

# print(f"The BRR weights are: \n {no_str_brr_wgt.head(20)} \n")


"""Balanced Repeated Replicates (BRR) WITH stratification"""
str_brr = ReplicateWeight(method="brr", stratification=True, number_reps=nb_reps, fay_coef=0.05)
str_brr_wgt = str_brr.replicate(sample_wgt, cluster_id, stratum_id)

# print(f"The BRR weights are: \n {str_brr_wgt.head(20)} \n")


"""Bootstrap replicates WITHOUT stratification"""
no_str_boot = ReplicateWeight(method="bootstrap", stratification=False, number_reps=nb_reps)
no_str_boot_wgt = no_str_boot.replicate(sample_wgt, cluster_id)

# print(f"The bootstrap weights are: \n {no_str_boot_wgt} \n")

"""Bootstrap replicates WITH stratification"""
str_boot = ReplicateWeight(method="bootstrap", stratification=True, number_reps=nb_reps)
str_boot_wgt = str_boot.replicate(sample_wgt, cluster_id, stratum_id)

# print(f"The stratified bootstrap weights are: \n {str_boot_wgt} \n")


"""Jackknife replicates WITHOUT stratification"""
no_str_jk = ReplicateWeight(method="jackknife", stratification=False)
no_str_jk_wgt = no_str_jk.replicate(sample_wgt, cluster_id)

# print(f"The jackknife weights are: \n {no_str_jk_wgt} \n")

"""Jackknife replicates WITH stratification"""
str_jk = ReplicateWeight(method="jackknife", stratification=True)
str_jk_wgt = str_jk.replicate(sample_wgt, cluster_id, stratum_id)

# print(f"The jackknife weights are: \n {str_jk_wgt} \n")
