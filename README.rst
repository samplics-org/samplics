==========
*SAMPLICS*
==========

*samplics* is a python package for selecting, weighting
and analyzing sample obtained from complex sampling design.

Sample Analytics
----------------
In large scale surveys, often complex random mechanisms are used to select
samples. Estimations obtained from such samples must reflect the random
mechanism to ensure accurate calculations. *samplics* implements a set of
sampling techniques for complex survey designs.

Selection
================
Since the full population cannot be observed, a sample is selected
to estimate population parameters of interest. The assumption is
that the sample is **representative** of the population for the characteristics
of interest. The selection methods in samplics are:

* Simple random sampling (SRS)
* Systematic selection (SYS)
* Probability proportional to size (PPS)
    * Systematic
    * Brewer's method
    * Hanurav-Vijayan method
    * Murphy's method
    * Sampford's method
* Unequal sample selection

Weighting
=========
Sample weighting is the main mechanism used in surveys to formalize the
representivity of the sample. The base or design weights are usually
adjusted to compensate for distorsions due nonresponse and other shorcomings
of the operationalization of the sampling design.

* Weight adjustment due to nonresponse
* Weight poststratification, calibration and normalization
* Weight replication i.e. Boostrap, BRR, and Jackknife

Estimation
==========
The estimation of the parameters of interest must reflect the sampling
mechanism and the weight adjustments.

 * Taylor linearization procedures
 * Replicate-based estimation i.e. Boostrap, BRR, and Jackknife
 * Regression-based

Parameters of interest
* Linear parameters e.g. total, mean, proportion
* Non-linear (complex) parameters e.g. ratio, regression coefficient

Installation
------------
``pip install samplics``

if both Python 2.x and python 3.x is installed on your computer,
you may have to use: ``pip3 install samplics``

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
    from samplics.sampling import Sample

    psu_frame = pd.read_csv("psu_frame.csv")
    psu_sample_size = {"East":3, "West": 2, "North": 2, "South": 3}
    pps_design = Sample(method="pps-sys", stratification=True, with_replacement=False)
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
        exclude_nan=True
    )

    ratio_wgt_hgt = ReplicateEstimator("brr", "ratio").estimate(
        y=nhanes2brr["weight"], 
        samp_weight=nhanes2brr["finalwgt"], 
        x=nhanes2brr["height"],
        rep_weights=nhanes2brr.loc[:, "brr_1":"brr_32"],
        exclude_nan = True
    )


Contributing
------------
TBD

License
-------
`MIT <https://github.com/survey-methods/samplics/blob/master/license.txt>`_

Project status
--------------
This is an alpha version. At this stage, this project is not recommended to be
used for production or any project that the user depend on.




