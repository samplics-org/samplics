==========
*SAMPLICS*
==========
.. image:: https://travis-ci.com/survey-methods/samplics.svg?token=WwRayqkQBt1W4ihyTzvw&branch=master
  :target: https://travis-ci.com/survey-methods/samplics

.. image:: https://codecov.io/gh/survey-methods/samplics/branch/master/graph/badge.svg?token=7C0LBB5N8Y
  :target: https://codecov.io/gh/survey-methods/samplics     

.. image:: https://readthedocs.org/projects/samplics/badge/?version=latest
  :target: https://samplics.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

*samplics* is a python package for selecting, weighting and analyzing sample obtained from complex sampling design.


In large scale surveys, often complex random mechanisms are used to select
samples. Estimations obtained from such samples must reflect the random
mechanism to ensure accurate calculations. *Samplics* implements a set of
sampling techniques for complex survey designs.

**Sampling.** Since the full population cannot be observed, a sample is selected
to estimate population parameters of interest. The assumption is
that the sample is representative of the population for the characteristics
of interest. The sample size calculation and selection methods in Samplics are:

* Sample size calculation and allocation: Wald and Fleiss methods for proportions. 
* Equal probability of selection: simple random sampling (SRS) and systematic selection (SYS)
* Probability proportional to size (PPS): Systematic, Brewer's method, Hanurav-Vijayan method, Murphy's method, and Rao-Sampford's method.

**Weighting.** Sample weighting is the main mechanism used in surveys to formalize the
representivity of the sample. The design/base weights are usually
adjusted to compensate for distortions due nonresponse and other shortcomings
of the the sampling design implementation.

* Weight adjustment due to nonresponse
* Weight poststratification, calibration and normalization
* Weight replication i.e. Bootstrap, BRR, and Jackknife

**Estimation.** The estimation of the parameters of interest must reflect the sampling
mechanism and the weight adjustments.

* Taylor-based procedures
* Replication-based estimation i.e. Boostrap, BRR, and Jackknife
* Regression-based e.g. generalized regression (GREG)

**Small Area Estimation (SAE).** When the sample size is not large enough to produce reliable / stable domain level estimates, SAE techniques can be used to modelled the output variable of interest to produce domain level estimaetes.

Installation
------------
``pip install samplics``

if both Python 2.x and python 3.x are installed on your computer, you may have to use: ``pip3 install samplics``

Dependencies
------------
Python versions 3.6.x or newer and the following packages:

* `numpy <https://numpy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `scpy <https://www.scipy.org/>`_
* `statsmodels <https://www.statsmodels.org/stable/index.h.tml>`_

Usage
------

To select a sample of primary sampling units using PPS method,
we can use a code similar to:

.. code:: python

    import samplics
    from samplics.sampling import SampleSelection

    psu_frame = pd.read_csv("psu_frame.csv")
    psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}
    pps_design = SampleSelection(method="pps-sys", stratification=True, with_replacement=False)
    frame["psu_prob"] = pps_design.inclusion_probs(
        psu_frame["cluster"],
        psu_sample_size,
        psu_frame["region"],
        psu_frame["number_households_census"]
        )

To adjust the design sample weight for nonresponse,
we can use a code similar to:

.. code:: python

    import samplics
    from samplics.weighting import SampleWeight

    status_mapping = {
        "in": "ineligible", "rr": "respondent", "nr": "non-respondent", "uk":"unknown"
        }

    full_sample["nr_weight"] = SampleWeight().adjust(
        samp_weight=full_sample["design_weight"],
        adjust_class=full_sample["region"],
        resp_status=full_sample["response_status"],
        resp_dict=status_mapping
        )

.. code:: python

    import samplics
    from samplics.estimation import TaylorEstimation, ReplicateEstimator

    zinc_mean_str = TaylorEstimator("mean").estimate(
        y=nhanes2f["zinc"],
        samp_weight=nhanes2f["finalwgt"],
        stratum=nhanes2f["stratid"],
        psu=nhanes2f["psuid"],
        remove_nan=True
    )

    ratio_wgt_hgt = ReplicateEstimator("brr", "ratio").estimate(
        y=nhanes2brr["weight"],
        samp_weight=nhanes2brr["finalwgt"],
        x=nhanes2brr["height"],
        rep_weights=nhanes2brr.loc[:, "brr_1":"brr_32"],
        remove_nan = True
    )


Contributing
------------
TBD

License
-------
`MIT <https://github.com/survey-methods/samplics/blob/master/license.txt>`_

Contact 
--------------
created by `Mamadou S. Diallo <https://twitter.com/MamadouSDiallo>`_ - feel free to contact me!




