get_ipython().run_cell_magic("capture", "", """%run psu_selection.ipynb""")


# Create a synthetic second stage frame
census_size = psu_frame.loc[psu_frame["psu_sample"] == 1, "number_households_census"].values
stratum_names = psu_frame.loc[psu_frame["psu_sample"] == 1, "region"].values
cluster = psu_frame.loc[psu_frame["psu_sample"] == 1, "cluster"].values

np.random.seed(15)

listing_size = np.zeros(census_size.size)
for k in range(census_size.size):
    listing_size[k] = np.random.normal(1.05 * census_size[k], 0.15 * census_size[k])

listing_size = listing_size.astype(int)
hh_id = rr_id = cl_id = []
for k, s in enumerate(listing_size):
    hh_k1 = np.char.array(np.repeat(stratum_names[k], s)).astype(str)
    hh_k2 = np.char.array(np.arange(1, s + 1)).astype(str)
    cl_k = np.repeat(cluster[k], s)
    hh_k = np.char.add(np.char.array(cl_k).astype(str), hh_k2)
    hh_id = np.append(hh_id, hh_k)
    rr_id = np.append(rr_id, hh_k1)
    cl_id = np.append(cl_id, cl_k)

ssu_frame = pd.DataFrame(cl_id.astype(int))
ssu_frame.rename(columns={0: "cluster"}, inplace=True)
ssu_frame["region"] = rr_id
ssu_frame["household"] = hh_id

nb_obs = 15
print(f"\nFirst {nb_obs} observations of the SSU frame\n")
ssu_frame.head(nb_obs)


psu_sample = psu_frame.loc[psu_frame["psu_sample"] == 1]
ssu_counts = ssu_frame.groupby("cluster").count()
ssu_counts.drop(columns="region", inplace=True)
ssu_counts.reset_index(inplace=True)
ssu_counts.rename(columns={"household": "number_households_listed"}, inplace=True)

pd.merge(
    psu_sample[["cluster", "region", "number_households_census"]],
    ssu_counts[["cluster", "number_households_listed"]],
    on=["cluster"],
)


stage2_design = Sample(method="srs", stratification=True, with_replacement=False)

ssu_frame["ssu_prob"] = stage2_design.inclusion_probs(ssu_frame["household"], 15, ssu_frame["cluster"])

ssu_frame.sample(20)


np.random.seed(11)
ssu_sample, ssu_hits, ssu_probs = stage2_design.select(ssu_frame["household"], 15, ssu_frame["cluster"])

ssu_frame["ssu_sample"] = ssu_sample
ssu_frame["ssu_hits"] = ssu_hits
ssu_frame["ssu_probs"] = ssu_probs

ssu_frame[ssu_frame["ssu_sample"] == 1].sample(15)


rates = np.repeat(15, 10) / ssu_counts["number_households_listed"].values

ssu_rates = dict(zip(np.unique(ssu_frame["cluster"]), rates))

ssu_rates


np.random.seed(22)

stage2_design2 = Sample(method="sys", stratification=True, with_replacement=False)

ssu_sample_r, ssu_hits_r, _ = stage2_design2.select(
    ssu_frame["household"], stratum=ssu_frame["cluster"], samp_rate=ssu_rates
)

ssu_sample2 = pd.DataFrame(
    data={"household": ssu_frame["household"], "ssu_sample_r": ssu_sample_r, "ssu_hits_r": ssu_hits_r}
)

ssu_sample2.head(25)


psu_sample[["cluster", "region", "psu_prob"]].to_csv("psu_sample.csv")

ssu_sample = ssu_frame.loc[ssu_frame["ssu_sample"] == 1]
ssu_sample[["cluster", "household", "ssu_prob"]].to_csv("ssu_sample.csv")
