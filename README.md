# <img src="./img/samplics-logo3.jpeg"> _SAMPLICS_

[![Build Status](https://travis-ci.com/survey-methods/samplics.svg?branch=master)](https://travis-ci.com/survey-methods/samplics/branches)
[![codecov](https://codecov.io/gh/survey-methods/samplics/branch/master/graph/badge.svg)](https://codecov.io/gh/survey-methods/samplics)
[![docs](https://readthedocs.org/projects/samplics/badge/?version=latest)](https://samplics.readthedocs.io/en/latest/?badge=latest)

In large scale surveys, often complex random mechanisms are used to select
samples. Estimates derived from such samples must reflect the random
mechanism. _Samplics_ is a python package that implements a set of
sampling techniques for complex survey designs. These survey sampling techniques are organized into the following four subpackages.

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

For more details, visit https://samplics.readthedocs.io/en/latest/

## Usage

Let's assume that we have a population and we would like to select a sample from it. The goal is to calculate the sample size for an expected proportion of 0.80 with a precision of 0.10.

> ```python
> import samplics
> from samplics.sampling import SampleSize
>
> sample_size = SampleSize(parameter = "proportion")
> sample_size.calculate(target=0.80, precision=0.10)
>
> assert size_nat_wald.samp_size["__none__"] == 62
> ```

Furthermore, the population is located in four natural regions i.e. North, South, East, and West. We could be interested in calculating sample sizes based on region specific requirements e.g. expected proportions, desired precisions and associated design effects.

> ```python
> import samplics
> from samplics.sampling import SampleSize
>
> sample_size = SampleSize(parameter="proportion", method="wald", stratification=True)
>
> expected_proportions = {"North": 0.95, "South": 0.70, "East": 0.30, "West": 0.50}
> half_ci = {"North": 0.30, "South": 0.10, "East": 0.15, "West": 0.10}
> deff = {"North": 1, "South": 1.5, "East": 2.5, "West": 2.0}
>
> sample_size = SampleSize(parameter = "proportion", method="Fleiss", stratification=True)
> sample_size.calculate(target=expected_proportions, precision=half_ci, deff=deff)
>
> assert size_nat_wald.samp_size["North"] == 11
> assert size_nat_wald.samp_size["South"] == 154
> assert size_nat_wald.samp_size["East"] == 115
> assert size_nat_wald.samp_size["West"] == 205
> ```

To select a sample of primary sampling units using PPS method,
we can use code similar to:

> ```python
> import samplics
> from samplics.sampling import SampleSelection
>
> psu_frame = pd.read_csv("psu_frame.csv")
> psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}
> pps_design = SampleSelection(
>    method="pps-sys",
>    stratification=True,
>    with_replacement=False
>    )
>
> frame["psu_prob"] = pps_design.inclusion_probs(
>    psu_frame["cluster"],
>    psu_sample_size,
>    psu_frame["region"],
>    psu_frame["number_households_census"]
>    )
> ```

To adjust the design sample weight for nonresponse,
we can use code similar to:

> ```python
> import samplics
> from samplics.weighting import SampleWeight
>
> status_mapping = {
>    "in": "ineligible",
>    "rr": "respondent",
>    "nr": "non-respondent",
>    "uk":"unknown"
>    }
>
> full_sample["nr_weight"] = SampleWeight().adjust(
>    samp_weight=full_sample["design_weight"],
>    adjust_class=full_sample["region"],
>    resp_status=full_sample["response_status"],
>    resp_dict=status_mapping
>    )
> ```

To estimate population parameters, we can use code similar to:

> ```python
> import samplics
> from samplics.estimation import TaylorEstimation, ReplicateEstimator
>
> # Taylor-based
> zinc_mean_str = TaylorEstimator("mean").estimate(
>    y=nhanes2f["zinc"],
>    samp_weight=nhanes2f["finalwgt"],
>    stratum=nhanes2f["stratid"],
>    psu=nhanes2f["psuid"],
>    remove_nan=True
> )
>
> # Replicate-based
> ratio_wgt_hgt = ReplicateEstimator("brr", "ratio").estimate(
>    y=nhanes2brr["weight"],
>    samp_weight=nhanes2brr["finalwgt"],
>    x=nhanes2brr["height"],
>    rep_weights=nhanes2brr.loc[:, "brr_1":"brr_32"],
>    remove_nan = True
> )
> ```

To predict small area parameters, we can use code similar to:

> ```python
> import samplics
> from samplics.estimation import EblupAreaModel, EblupUnitModel
>
> # Area-level basic method
> fh_model_reml = EblupAreaModel(method="REML")
> fh_model_reml.fit(
>    yhat=yhat, X=X, area=area, intercept=False, error_std=sigma_e, tol=1e-4,
> )
> fh_model_reml.predict(X=X, area=area, intercept=False)
>
> # Unit-level basic method
> eblup_bhf_reml = EblupUnitModel()
> eblup_bhf_reml.fit(ys, Xs, areas,)
> eblup_bhf_reml.predict(Xmean, areas_list)
> ```

## Installation

`pip install samplics`

Python 3.6 or newer is required and the main dependencies are [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [scpy](https://www.scipy.org), and [statsmodel](https://www.statsmodels.org/stable/index.h.tml).

## License

[MIT](https://github.com/survey-methods/samplics/blob/master/license.txt)

## Contact

created by [Mamadou S. Diallo](https://twitter.com/MamadouSDiallo) - feel free to contact me!
