<img src="./img/samplics_logo.jpg"  align="left" style="height: 110px; border-radius: 10%; padding: 5px;"/>


<h1> Sample Analytics </h1>

[<img src="https://github.com/survey-methods/samplics/workflows/Testing/badge.svg">](https://github.com/survey-methods/samplics/actions?query=workflow%3ATesting)
[<img src="https://github.com/survey-methods/samplics/workflows/Coverage/badge.svg">](https://github.com/survey-methods/samplics/actions?query=workflow%3ACoverage)
[<img src="https://github.com/survey-methods/samplics/workflows/Docs/badge.svg">](https://github.com/samplics-org/samplics/actions?query=workflow%3ADocs)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03376/status.svg)](https://doi.org/10.21105/joss.03376)
[<img src="https://pepy.tech/badge/samplics">](https://pepy.tech/project/samplics)



In large-scale surveys, often complex random mechanisms are used to select samples. Estimates derived from such samples must reflect the random mechanism. _Samplics_ is a python package that implements a set of sampling techniques for complex survey designs. These survey sampling techniques are organized into the following four sub-packages.

**Sampling** provides a set of random selection techniques used to draw a sample from a population. It also provides procedures for calculating sample sizes. The sampling subpackage contains:

- Sample size calculation and allocation: Wald and Fleiss methods for proportions.
- Equal probability of selection: simple random sampling (SRS) and systematic selection (SYS)
- Probability proportional to size (PPS): Systematic, Brewer's method, Hanurav-Vijayan method, Murphy's method, and Rao-Sampford's method.

**Weighting** provides the procedures for adjusting sample weights. More specifically, the weighting subpackage allows the following:

- Weight adjustment due to nonresponse
- Weight poststratification, calibration and normalization
- Weight replication i.e. Bootstrap, BRR, and Jackknife

**Estimation** provides methods for estimating the parameters of interest with uncertainty measures that are consistent with the sampling design. The estimation subpackage implements the following types of estimation methods:

- Taylor-based, also called linearization methods
- Replication-based estimation i.e. Boostrap, BRR, and Jackknife
- Regression-based e.g. generalized regression (GREG)

**Small Area Estimation (SAE).** When the sample size is not large enough to produce reliable / stable domain level estimates, SAE techniques can be used to model the output variable of interest to produce domain level estimates. This subpackage provides Area-level and Unit-level SAE methods.

For more details, visit https://samplics-org.github.io/samplics/

## Usage

Let's assume that we have a population and we would like to select a sample from it. The goal is to calculate the sample size for an expected proportion of 0.80 with a precision (half confidence interval) of 0.10.

> ```python
> from samplics.sampling import SampleSize
>
> sample_size = SampleSize(parameter = "proportion")
> sample_size.calculate(target=0.80, half_ci=0.10)
> ```

Furthermore, the population is located in four natural regions i.e. North, South, East, and West. We could be interested in calculating sample sizes based on region specific requirements e.g. expected proportions, desired precisions and associated design effects.

> ```python
> from samplics.sampling import SampleSize
>
> sample_size = SampleSize(parameter="proportion", method="wald", strat=True)
>
> expected_proportions = {"North": 0.95, "South": 0.70, "East": 0.30, "West": 0.50}
> half_ci = {"North": 0.30, "South": 0.10, "East": 0.15, "West": 0.10}
> deff = {"North": 1, "South": 1.5, "East": 2.5, "West": 2.0}
>
> sample_size = SampleSize(parameter = "proportion", method="Fleiss", strat=True)
> sample_size.calculate(target=expected_proportions, half_ci=half_ci, deff=deff)
> ```

To select a sample of primary sampling units using PPS method,
we can use code similar to the snippets below. Note that we first use the `datasets` module to import the example dataset.

> ```python
> # First we import the example dataset
> from samplics.datasets import load_psu_frame
> psu_frame_dict = load_psu_frame()
> psu_frame = psu_frame_dict["data"]
>
> # Code for the sample selection
> from samplics.sampling import SampleSelection
> from samplics.utils import SelectMethod
>
> psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}
> pps_design = SampleSelection(
>    method=SelectMethod.pps_sys,
>    strat=True,
>    wr=False
>    )
>
> psu_frame["psu_prob"] = pps_design.inclusion_probs(
>    psu_frame["cluster"],
>    psu_sample_size,
>    psu_frame["region"],
>    psu_frame["number_households_census"]
>    )
> ```

The initial weighting step is to obtain the design sample weights. In this example, we show a simple example of two-stage sampling design.

> ```python
> import pandas as pd
>
> from samplics.datasets import load_psu_sample, load_ssu_sample
> from samplics.weighting import SampleWeight
>
> # Load PSU sample data
> psu_sample_dict = load_psu_sample()
> psu_sample = psu_sample_dict["data"]
>
> # Load PSU sample data
> ssu_sample_dict = load_ssu_sample()
> ssu_sample = ssu_sample_dict["data"]
>
> full_sample = pd.merge(
>     psu_sample[["cluster", "region", "psu_prob"]],
>     ssu_sample[["cluster", "household", "ssu_prob"]],
>     on="cluster"
> )
>
> full_sample["inclusion_prob"] = full_sample["psu_prob"] * full_sample["ssu_prob"]
> full_sample["design_weight"] = 1 / full_sample["inclusion_prob"]
> ```

To adjust the design sample weight for nonresponse,
we can use code similar to:

> ```python
> import numpy as np
>
> from samplics.weighting import SampleWeight
>
> # Simulate response
> np.random.seed(7)
> full_sample["response_status"] = np.random.choice(
>     ["ineligible", "respondent", "non-respondent", "unknown"],
>     size=full_sample.shape[0],
>     p=(0.10, 0.70, 0.15, 0.05),
> )
> # Map custom response statuses to teh generic samplics statuses
> status_mapping = {
>    "in": "ineligible",
>    "rr": "respondent",
>    "nr": "non-respondent",
>    "uk":"unknown"
>    }
> # adjust sample weights
> full_sample["nr_weight"] = SampleWeight().adjust(
>    samp_weight=full_sample["design_weight"],
>    adjust_class=full_sample["region"],
>    resp_status=full_sample["response_status"],
>    resp_dict=status_mapping
>    )
> ```

To estimate population parameters using Taylor-based and replication-based methods, we can use code similar to:

> ```python
> # Taylor-based
> from samplics.datasets import load_nhanes2
>
> nhanes2_dict = load_nhanes2()
> nhanes2 = nhanes2_dict["data"]
>
> from samplics.estimation import TaylorEstimator
>
> zinc_mean_str = TaylorEstimator("mean")
> zinc_mean_str.estimate(
>     y=nhanes2["zinc"],
>     samp_weight=nhanes2["finalwgt"],
>     stratum=nhanes2["stratid"],
>     psu=nhanes2["psuid"],
>     remove_nan=True,
> )
>
> # Replicate-based
> from samplics.datasets import load_nhanes2brr
>
> nhanes2brr_dict = load_nhanes2brr()
> nhanes2brr = nhanes2brr_dict["data"]
>
> from samplics.estimation import ReplicateEstimator
>
> ratio_wgt_hgt = ReplicateEstimator("brr", "ratio").estimate(
>     y=nhanes2brr["weight"],
>     samp_weight=nhanes2brr["finalwgt"],
>     x=nhanes2brr["height"],
>     rep_weights=nhanes2brr.loc[:, "brr_1":"brr_32"],
>     remove_nan=True,
> )
>
> ```

To predict small area parameters, we can use code similar to:

> ```python
> import numpy as np
> import pandas as pd
>
> # Area-level basic method
> from samplics.datasets import load_expenditure_milk
>
> milk_exp_dict = load_expenditure_milk()
> milk_exp = milk_exp_dict["data"]
>
> from samplics.sae import EblupAreaModel
>
> fh_model_reml = EblupAreaModel(method="REML")
> fh_model_reml.fit(
>     yhat=milk_exp["direct_est"],
>     X=pd.get_dummies(milk_exp["major_area"], drop_first=True),
>     area=milk_exp["small_area"],
>     error_std=milk_exp["std_error"],
>     intercept=True,
>     tol=1e-8,
> )
> fh_model_reml.predict(
>     X=pd.get_dummies(milk_exp["major_area"], drop_first=True),
>     area=milk_exp["small_area"],
>     intercept=True,
> )
>
> # Unit-level basic method
> from samplics.datasets import load_county_crop, load_county_crop_means
>
> # Load County Crop sample data
> countycrop_dict = load_county_crop()
> countycrop = countycrop_dict["data"]
> # Load County Crop Area Means sample data
> countycropmeans_dict = load_county_crop_means()
> countycrop_means = countycropmeans_dict["data"]
>
> from samplics.sae import EblupUnitModel
>
> eblup_bhf_reml = EblupUnitModel()
> eblup_bhf_reml.fit(
>     countycrop["corn_area"],
>     countycrop[["corn_pixel", "soybeans_pixel"]],
>     countycrop["county_id"],
> )
> eblup_bhf_reml.predict(
>     Xmean=countycrop_means[["ave_corn_pixel", "ave_corn_pixel"]],
>     area=np.linspace(1, 12, 12),
> )
>
> ```

## Installation

`pip install samplics`

Python 3.7 or newer is required and the main dependencies are [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [scpy](https://www.scipy.org), and [statsmodel](https://www.statsmodels.org/stable/index.html).

## Contribution

If you would like to contribute to the project, please read [contributing to samplics](https://github.com/samplics-org/samplics/blob/main/CONTRIBUTING.md)

## License

[MIT](https://github.com/survey-methods/samplics/blob/master/license.txt)

## Contact

created by [Mamadou S. Diallo](https://twitter.com/MamadouSDiallo) - feel free to contact me!
