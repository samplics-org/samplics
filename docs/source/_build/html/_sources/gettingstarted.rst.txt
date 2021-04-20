==================
How to get started
==================

*samplics* is a python package for selecting, weighting and analyzing sample obtained from complex sampling design.


Installation
============
``pip install samplics``

if both Python 2.x and python 3.x are installed on your computer, you may have to use: ``pip3 install samplics``

Dependencies
============
Python versions 3.7.x or newer and the following packages:

* `numpy <https://numpy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `scpy <https://www.scipy.org/>`_
* `statsmodels <https://www.statsmodels.org/stable/index.h.tml>`_

Usage
=====

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
============
TBD

License
=======
`MIT <https://github.com/survey-methods/samplics/blob/master/license.txt>`_






