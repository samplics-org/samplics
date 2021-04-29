import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

acs2019 = pd.read_csv("~/data/IPUMS/acs/2019/acs2019_00001_v2.csv")

print(f"Size of the ACS2019 file is: {acs2019.shape}")

acs2019_head_hh = acs2019.loc[acs2019["relate"].isin([1]) & acs2019["gq"].isin([1, 2, 5])]

print(f"Size of the ACS2019 head of HH file is: {acs2019_head_hh.shape}")
# acs2019_head_hh.to_csv("./acs2019_head_hh.csv")

psu = acs2019_head_hh[["stateicp", "countyicp", "puma", "city"]]
_, nb_hhs = np.unique(
    psu,
    return_counts=True,
    axis=0,
)
print(nb_hhs.shape)

psuid = psu.drop_duplicates(ignore_index=True)
psuid["psu"] = np.linspace(1, nb_hhs.shape[0], num=nb_hhs.shape[0]).astype(int)
# print(psuid.columns)

acs2019_head_hh = acs2019_head_hh.merge(psuid, on=["stateicp", "countyicp", "puma", "city"])
acs2019_head_hh["hhid"] = np.linspace(
    1, acs2019_head_hh.shape[0], num=acs2019_head_hh.shape[0]
).astype(int)
print(acs2019_head_hh.columns)

acs2019_head_hh["region"] = "NA"
acs2019_head_hh = acs2019_head_hh[
    [
        "hhid",
        "region",
        "stateicp",
        "psu",
        "sex",
        "age",
        "marst",
        "race",
        "citizen",
        "speakeng",
        "educ",
        "foodstmp",
        "famsize",
        "ftotinc",
    ]
]

acs2019_head_hh.sex = acs2019_head_hh.sex.map({1: "Male", 2: "Female"})
acs2019_head_hh.educ = acs2019_head_hh.educ.map(
    {
        0: "No schooling",
        1: "No college",
        2: "No college",
        3: "No college",
        4: "No college",
        5: "No college",
        6: "No college",
        7: "College",
        8: "College",
        9: "College",
        10: "College",
        11: "College",
    }
)
acs2019_head_hh.speakeng = acs2019_head_hh.speakeng.map(
    {
        0: "Blank",
        1: "No",
        2: "Yes",
        3: "Yes",
        4: "Yes",
        5: "Yes",
        6: "Yes",
        7: "Unknown",
        8: "Ineligible",
    }
)
acs2019_head_hh.marst = acs2019_head_hh.marst.map(
    {
        1: "Married",
        2: "Married",
        3: "Separated",
        4: "Divorced",
        5: "Widowed",
        6: "Single",
    }
)
acs2019_head_hh.race = acs2019_head_hh.race.map(
    {
        1: "White",
        2: "Black",
        3: "Indian/Native",
        4: "Asian",
        5: "Asian",
        6: "Asian",
        7: "Other",
        8: "Mixed",
        9: "Mixed",
    }
)
acs2019_head_hh.citizen = acs2019_head_hh.citizen.map(
    {
        0: "Citizen",
        1: "Citizen",
        2: "Citizen",
        3: "Not a citizen",
        4: "Not a citizen",
        5: "Don't know",
    }
)

acs2019_head_hh["region"] = "South"
acs2019_head_hh.loc[
    acs2019_head_hh.stateicp.isin([1, 2, 3, 4, 5, 6, 12, 13, 14]), "region"
] = "Northest"
acs2019_head_hh.loc[
    acs2019_head_hh.stateicp.isin([21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37]), "region"
] = "Midwest"
acs2019_head_hh.loc[
    acs2019_head_hh.stateicp.isin([61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 81, 82]),
    "region",
] = "West"

acs2019_head_hh.rename(
    {
        "stateicp": "state",
        "marst": "marital_status",
        "speakeng": "speak_english",
        "educ": "education",
        "foodstmp": "food_stamp",
        "famsize": "family_size",
        "ftotinc": "family_income",
    },
    inplace=True,
    axis="columns",
)

acs2019_head_hh.loc[acs2019_head_hh.state == 1, "state"] = "Connecticut"
acs2019_head_hh.loc[acs2019_head_hh.state == 2, "state"] = "Maine"
acs2019_head_hh.loc[acs2019_head_hh.state == 3, "state"] = "Massachusetts"
acs2019_head_hh.loc[acs2019_head_hh.state == 4, "state"] = "New Hampshire"
acs2019_head_hh.loc[acs2019_head_hh.state == 5, "state"] = "Rhode Island"
acs2019_head_hh.loc[acs2019_head_hh.state == 6, "state"] = "Vermont"

acs2019_head_hh.loc[acs2019_head_hh.state == 11, "state"] = "Delaware"
acs2019_head_hh.loc[acs2019_head_hh.state == 12, "state"] = "New Jersey"
acs2019_head_hh.loc[acs2019_head_hh.state == 13, "state"] = "New York"
acs2019_head_hh.loc[acs2019_head_hh.state == 14, "state"] = "Pennsylvania"

acs2019_head_hh.loc[acs2019_head_hh.state == 21, "state"] = "Illinois"
acs2019_head_hh.loc[acs2019_head_hh.state == 22, "state"] = "Indiana"
acs2019_head_hh.loc[acs2019_head_hh.state == 23, "state"] = "Michigan"
acs2019_head_hh.loc[acs2019_head_hh.state == 24, "state"] = "Ohio"
acs2019_head_hh.loc[acs2019_head_hh.state == 25, "state"] = "Wisconsin"

acs2019_head_hh.loc[acs2019_head_hh.state == 31, "state"] = "Iowa"
acs2019_head_hh.loc[acs2019_head_hh.state == 32, "state"] = "Kansas"
acs2019_head_hh.loc[acs2019_head_hh.state == 33, "state"] = "Minnesota"
acs2019_head_hh.loc[acs2019_head_hh.state == 34, "state"] = "Missouri"
acs2019_head_hh.loc[acs2019_head_hh.state == 35, "state"] = "Nebraska"
acs2019_head_hh.loc[acs2019_head_hh.state == 36, "state"] = "North Dakota"
acs2019_head_hh.loc[acs2019_head_hh.state == 37, "state"] = "South Dakota"

acs2019_head_hh.loc[acs2019_head_hh.state == 40, "state"] = "Virginia"
acs2019_head_hh.loc[acs2019_head_hh.state == 41, "state"] = "Alabama"
acs2019_head_hh.loc[acs2019_head_hh.state == 42, "state"] = "Arkansas"
acs2019_head_hh.loc[acs2019_head_hh.state == 43, "state"] = "Florida"
acs2019_head_hh.loc[acs2019_head_hh.state == 44, "state"] = "Georgia"
acs2019_head_hh.loc[acs2019_head_hh.state == 45, "state"] = "Louisiana"
acs2019_head_hh.loc[acs2019_head_hh.state == 46, "state"] = "Mississipi"
acs2019_head_hh.loc[acs2019_head_hh.state == 47, "state"] = "North Carolina"
acs2019_head_hh.loc[acs2019_head_hh.state == 48, "state"] = "South Carolina"
acs2019_head_hh.loc[acs2019_head_hh.state == 49, "state"] = "Texas"

acs2019_head_hh.loc[acs2019_head_hh.state == 51, "state"] = "Kentucky"
acs2019_head_hh.loc[acs2019_head_hh.state == 52, "state"] = "Maryland"
acs2019_head_hh.loc[acs2019_head_hh.state == 53, "state"] = "Oklahoma"
acs2019_head_hh.loc[acs2019_head_hh.state == 54, "state"] = "Tennessee"
acs2019_head_hh.loc[acs2019_head_hh.state == 56, "state"] = "West Virginia"

acs2019_head_hh.loc[acs2019_head_hh.state == 61, "state"] = "Arizona"
acs2019_head_hh.loc[acs2019_head_hh.state == 62, "state"] = "Colorado"
acs2019_head_hh.loc[acs2019_head_hh.state == 63, "state"] = "Idaho"
acs2019_head_hh.loc[acs2019_head_hh.state == 64, "state"] = "Montana"
acs2019_head_hh.loc[acs2019_head_hh.state == 65, "state"] = "Nevada"
acs2019_head_hh.loc[acs2019_head_hh.state == 66, "state"] = "New Mexico"
acs2019_head_hh.loc[acs2019_head_hh.state == 67, "state"] = "Utah"
acs2019_head_hh.loc[acs2019_head_hh.state == 68, "state"] = "Wyoming"

acs2019_head_hh.loc[acs2019_head_hh.state == 71, "state"] = "California"
acs2019_head_hh.loc[acs2019_head_hh.state == 72, "state"] = "Oregon"
acs2019_head_hh.loc[acs2019_head_hh.state == 73, "state"] = "Washington"

acs2019_head_hh.loc[acs2019_head_hh.state == 81, "state"] = "Alaska"
acs2019_head_hh.loc[acs2019_head_hh.state == 82, "state"] = "Hawaii"
acs2019_head_hh.loc[acs2019_head_hh.state == 83, "state"] = "Puerto Rico"

acs2019_head_hh.loc[acs2019_head_hh.state == 96, "state"] = "State groupings"
acs2019_head_hh.loc[acs2019_head_hh.state == 97, "state"] = "Overseas Military"
acs2019_head_hh.loc[acs2019_head_hh.state == 98, "state"] = "District of Columbia"
acs2019_head_hh.loc[acs2019_head_hh.state == 99, "state"] = "State not identified"


print(acs2019_head_hh.head())

acs2019_head_hh.to_csv("./acs2019_head_hh.csv", index=False)

# nb_pd = pd.DataFrame(nb_hhs[1], columns=["nb_hhs"])
# sns.displot(data=nb_pd, x="nb_hhs", kde=True)