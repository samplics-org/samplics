==========================
SAMPLICS: sample analytics
==========================
.. image:: ![CI](https://github.com/survey-methods/samplics/tree/main/.github/workflows/Testing-Samplics/badge.svg?branch=master)
  :target: https://travis-ci.com/survey-methods/samplics

.. image:: https://codecov.io/gh/survey-methods/samplics/branch/master/graph/badge.svg?token=7C0LBB5N8Y
  :target: https://codecov.io/gh/survey-methods/samplics  

.. image:: https://readthedocs.org/projects/samplics/badge/?version=latest
  :target: https://samplics.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
         
In large scale surveys, often complex random mechanisms are used to select
samples. Estimations obtained from such samples must reflect the random
mechanism to ensure accurate calculations. *samplics* implements a set of
sampling techniques for complex survey designs.

**Sampling.** Since the full population cannot be observed, a sample is selected
to estimate population parameters of interest. The assumption is
that the sample is representative of the population for the characteristics
of interest. The sample size calculation and selection methods in *samplics* are:

* Sample size calculation and allocation: Wald and Fleiss methods for proportions. 
* Equal probability of selection: simple random sampling (SRS) and systematic selection (SYS)
* Probability proportional to size (PPS): Systematic, Brewer's method, Hanurav-Vijayan method, Murphy's method, and Rao-Sampford's method.

**Sample weighting.** Sample weighting is the main mechanism used in surveys to formalize the
representivity of the sample. The design/base weights are usually
adjusted to compensate for distortions due nonresponse and other shortcomings
of the the sampling design implementation.

* Weight adjustment due to nonresponse
* Weight poststratification, calibration and normalization
* Weight replication i.e. Bootstrap, BRR, and Jackknife

**Population parameters estimation.** The estimation of the parameters of interest must reflect the sampling
mechanism and the weight adjustments.

* Taylor-based procedures
* Replication-based estimation i.e. Boostrap, BRR, and Jackknife
* Regression-based e.g. generalized regression (GREG)

**Categorical data analysis.** The estimation of the parameters of interest must reflect the sampling
mechanism and the weight adjustments.

* one-way and two-way tabulation
* T-test including Rao-Scott adjustment

**Small Area Estimation (SAE).** When the sample size is not large enough to produce reliable / stable domain level estimates, SAE techniques can be used to model the output variable of interest to produce domain level estimates.

Documentation
#############

.. toctree::
   :maxdepth: 1

   gettingstarted
   tutorial/index
   usecases

   modules 

Index and search
################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**Contact Information**
| `Mamadou S. Diallo <https://twitter.com/MamadouSDiallo>`_
| msdiallo@samplics.org