import numpy as np


population_size = 35000000

admin1_nb = 10
admin1_share = np.array(
    [0.005, 0.020, 0.045, 0.075, 0.095, 0.105, 0.125, 0.130, 0.150, 0.250]
)
admin1_size = admin1_share * population_size

if sum(admin1_share) != 1.000:
    raise AssertionError("The admin level 1 shares must sum to 1")

# admin1 = np.random.choice(
#     a=np.linspace(1, admin1_nb, admin1_nb, dtype="int8"), size=population_size, p=admin1_share
# )

admin2_nb = 45
admin2_nb_by_admnin1 = np.array([1, 1, 2, 3, 5, 6, 5, 7, 10, 5])
admin2_share_1 = np.array([1]) * admin1_share[0]
admin2_share_2 = np.array([1]) * admin1_share[1]
admin2_share_3 = np.array([0.3, 0.7]) * admin1_share[2]
admin2_share_4 = np.array([0.4, 0.4, 0.2]) * admin1_share[3]
admin2_share_5 = (np.ones(5) / 5) * admin1_share[4]
admin2_share_6 = (np.ones(6) / 6) * admin1_share[5]
admin2_share_7 = np.linspace(1, 10, 5) / sum(np.linspace(1, 10, 5)) * admin1_share[6]
admin2_share_8 = np.linspace(1, 10, 7) / sum(np.linspace(1, 10, 7)) * admin1_share[7]
admin2_share_9 = np.linspace(1, 10, 10) / sum(np.linspace(1, 10, 10)) * admin1_share[8]
admin2_share_10 = np.linspace(1, 10, 5) / sum(np.linspace(1, 10, 5)) * admin1_share[9]
admin2_share = np.concatenate(
    (
        admin2_share_1,
        admin2_share_2,
        admin2_share_3,
        admin2_share_4,
        admin2_share_5,
        admin2_share_6,
        admin2_share_7,
        admin2_share_8,
        admin2_share_9,
        admin2_share_10,
    )
)

admin2_share = admin2_share / sum(admin2_share)

admin2 = np.random.choice(
    a=np.linspace(1, admin2_nb, admin2_nb, dtype="int8"),
    size=population_size,
    p=admin2_share,
)

_, size2 = np.unique(admin2, return_counts=True)

# print(size2 / population_size)
# print(admin2)

number_admin3 = 120  # equivalent to health disctrict for this use case
number_admin4 = 550
number_admin5 = 1250

# proportion_female = 0.55
female_age_distribution_urban = {
    "0-4": 7.3,
    "5-9": 6.6,
    "10-14": 5.8,
    "15-19": 5.1,
    "20-24": 4.4,
    "25-29": 3.9,
    "30-34": 3.5,
    "35-39": 3.0,
    "40-44": 2.4,
    "45-49": 1.9,
    "50-54": 1.6,
    "55-59": 1.3,
    "60-64": 1.1,
    "65-69": 0.8,
    "70-74": 0.6,
    "75-79": 0.3,
    "80-84": 0.2,
    "85-89": 0.06,
    "90-94": 0.03,
    "95-99": 0.01,
    "100+": 0.0,
}
male_age_distribution_urban = {
    "0-4": 7.5,
    "5-9": 6.8,
    "10-14": 6.0,
    "15-19": 5.2,
    "20-24": 4.5,
    "25-29": 3.9,
    "30-34": 3.5,
    "35-39": 2.9,
    "40-44": 2.4,
    "45-49": 1.9,
    "50-54": 1.5,
    "55-59": 1.2,
    "60-64": 1.0,
    "65-69": 0.7,
    "70-74": 0.5,
    "75-79": 0.3,
    "80-84": 0.1,
    "85-89": 0.1,
    "90-94": 0.05,
    "95-99": 0.03,
    "100+": 0.02,
}

female_age_distribution_rural = {
    "0-4": 7.3,
    "5-9": 6.6,
    "10-14": 5.8,
    "15-19": 5.1,
    "20-24": 4.4,
    "25-29": 3.9,
    "30-34": 3.5,
    "35-39": 3.0,
    "40-44": 2.4,
    "45-49": 1.9,
    "50-54": 1.6,
    "55-59": 1.3,
    "60-64": 1.1,
    "65-69": 0.8,
    "70-74": 0.6,
    "75-79": 0.3,
    "80-84": 0.2,
    "85-89": 0.06,
    "90-94": 0.03,
    "95-99": 0.01,
    "100+": 0.0,
}

male_age_distribution_rural = {
    "0-4": 7.5,
    "5-9": 6.8,
    "10-14": 6.0,
    "15-19": 5.2,
    "20-24": 4.5,
    "25-29": 3.9,
    "30-34": 3.5,
    "35-39": 2.9,
    "40-44": 2.4,
    "45-49": 1.9,
    "50-54": 1.5,
    "55-59": 1.2,
    "60-64": 1.0,
    "65-69": 0.7,
    "70-74": 0.5,
    "75-79": 0.3,
    "80-84": 0.1,
    "85-89": 0.1,
    "90-94": 0.05,
    "95-99": 0.03,
    "100+": 0.02,
}
# print(sum(list(female_age_distribution.values())))
# print(sum(list(male_age_distribution.values())))

# print(np.random.choice((1, 2, 3), size=150000, p=(0.1, 0.2, 0.7)))
