import numpy as np
import pandas as pd

# from scipy.stats import gamma

np.random.seed(181336)

number_regions = 5
number_strata = 10
number_units = 5000

units = np.linspace(0, number_units - 1, number_units, dtype="int16") + 10 * number_units
units = units.astype("str")

sample = pd.DataFrame(units)
sample.rename(columns={0: "unit_id"}, inplace=True)

sample["region_id"] = "xx"
for i in range(number_units):
    sample.loc[i]["region_id"] = sample.iloc[i]["unit_id"][0:2]

sample["cluster_id"] = "xxx"
for i in range(number_units):
    sample.loc[i]["cluster_id"] = sample.iloc[i]["unit_id"][0:4]

area_type = pd.DataFrame(np.unique(sample["cluster_id"]))
area_type.rename(columns={0: "cluster_id"}, inplace=True)
area_type["area_type"] = np.random.choice(("urban", "rural"), area_type.shape[0], p=(0.4, 0.6))
sample = pd.merge(sample, area_type, on="cluster_id")

sample["response_status"] = np.random.choice(
    ("IN", "RR", "NR", "UK"), number_units, p=(0.01, 0.81, 0.15, 0.03)
)
print(pd.crosstab(sample["region_id"], sample["response_status"]))

sample["educ_level"] = np.random.choice(
    ("0. Primary", "1. High-School", "2. University"), number_units, p=(0.10, 0.60, 0.30),
)
print(pd.crosstab(sample["region_id"], sample["educ_level"]))

sample["income"] = 0
low_income = 30000
primary = sample["educ_level"] == "0. Primary"
sample.income[primary] = np.random.gamma(low_income / 30e3, 30e3, np.sum(primary))

middle_income = 60000
highschool = sample["educ_level"] == "1. High-School"
sample.income[highschool] = np.random.gamma(
    middle_income / 10e3, 10e3, np.sum(highschool)
) + np.random.normal(low_income / 4, low_income / 4, np.sum(highschool))

high_income = 90000
university = sample["educ_level"] == "2. University"
sample.income[university] = np.random.gamma(
    high_income / 15e3, 15e3, np.sum(university)
) + np.random.normal(low_income / 2, low_income / 2, np.sum(university))

sample["income_level"] = "middle"
sample.loc[sample["income"] >= 100000, "income_level"] = "high"
sample.loc[sample["income"] <= 50000, "income_level"] = "low"
# print(pd.crosstab(sample["region_id"], sample["income_level"]))
# print(pd.crosstab(sample["educ_level"], sample["income_level"]))

sample.loc[sample["response_status"] == "NR", "income"] = np.nan
sample.loc[sample["response_status"] == "UK", "income"] = np.nan

sample.loc[sample["response_status"] == "NR", "income_level"] = ""
sample.loc[sample["response_status"] == "UK", "income_level"] = ""

sample.loc[sample["response_status"] == "NR", "educ_level"] = ""
sample.loc[sample["response_status"] == "UK", "educ_level"] = ""

# print(sample[primary])
# print(sample[highschool])
# print(sample[university])

sample["design_wgt"] = np.round(sample["cluster_id"].astype("int") / 10, 0)

# print(sum(sample["design_wgt"]))
# print(sample.sample(25))
print(
    sample[
        [
            "region_id",
            "cluster_id",
            "area_type",
            "response_status",
            "educ_level",
            "income",
            "design_wgt",
        ]
    ].sample(50)
)


sample.to_csv("./tests/weighting/synthetic_income_data.csv")
