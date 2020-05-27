
Tutorial
========

Samplics is a Python package designed to be a comprehensive tool for selecting, weighting and analyzing survey sample data obtained from complex designs. The main objective of the tutorial is to take the user through the Samplics' APIs main features. We hope that after going through the tutorial, the user will have a good understanding of the APIs' use cases. The tutorial is organized in several main topics. The user should be able to only focus on the topics of interest. 

This tutorial is intended to users with basic knowledge of Python. It is assumed that the user has a basic understanding of the Python syntax. Also, the tutorial is not intended to teach survey sampling methods. To learn survey sampling methods, we refer the user to the reference material mentioned throughout this tutorial. 


Sampling
--------

When a sample frame is available, the simple random sampling (srs) method is usually the easiest way to select a sample. However, operational and cost constraints often result in much more complex mechanisms of selecting samples. A common use case of complex sampling designs is the national household surveys such as the `Demographic and Health Surveys (DHS) <https://dhsprogram.com>`_, the `Living Standards Measurement Study (LSMS) <http://surveys.worldbank.org>`_, the `Multiple Indicator Cluster Surveys (MICS) <https://mics.unicef.org>`_, and many more. In the national household surveys, the sample is selected in stages. For example, at the first stage clusters of households are selected using probability proportional to size (pps) method, at the second stage households are selected from the sampled clusters, and at the final stage individuals are selected from households in the sample.  

In this tutorial, we generate simulated data to illustrate the first and second stage selections of clusters and households, respectively. 

.. toctree::
   :maxdepth: 1
   
   psu_selection
   ssu_selection

For a comprehensive review of complex sampling techniques, users may want to consult Brewer, K.R.W. and Hanif, M. (1983) [#bh1983]_, Cochran, W.G. (1977) [#c1977]_, Kish, L. (1965) [#k1965]_, and Lohr, S.L. (2010) [#l2010]_.

.. [#bh1983] Brewer, K.R.W. and Hanif, M. (1983), *Sampling With Unequal Probabilities*,  
   Springer-Verlag New York, Inc
.. [#c1977] Cochran, W.G. (1977), *Sampling Techniques, 3rd edn.*, Jonh Wiley & Sons, Inc.
.. [#k1965] Kish, L. (1965), *Survey Sampling*, Jonh Wiley & Sons, Inc.
.. [#l2010] Lohr, S.L. (2010), *Sampling: Design and Analysis, 2nd edn.*, Cengage Learning, Inc.

Weighting
---------

The sample weighting is the mechanism that allows us to generalize the results from the sample to the target population. The design/base weights are obtained from the final probability of selection as their inverse. In large scale surveys, design weights are adjusted to account for non-response, for extreme values, or to ensure that auxiliary variables benchmark to known controls. 

The weighting tutorial is in two parts. The first part discuss several weight adjustments methods e.g. non-response adjustment, poststratification, calibration. The second part of the tutorial walks the user through the creation of replicates weights using Booststrap, Balanced Repeated Replication (BRR), and Jackknife methods. 

.. toctree::
   :maxdepth: 1
   
  sample_weights
  replicate_weights

Valliant, R. and Dever, J. A. (2018) [#vd2018]_ provides a step-by-step guide on calculating 
sample weights.

.. [#vd2018] Valliant, R. and Dever, J. A. (2018), *Survey Weights: A Step-by-Step Guide to       
   Calculation*, Stata Press.

Estimation
----------

Often the goal of conducting a survey is to estimate parameters of the target population from the survey sample. To appropriately estimate uncertainties associated with the sample estimates, the complex design must be taken into account. In this tutorial, we show how to use the APIs to produce point estimates of mean, total, proportion and ratio parameters as well as their associated Taylor-based and replication-based measures of uncertainty. Wolter, K.M. (2007) [#w2007]_ provides a self-contained description of a number of techniques for variance estimation both Taylor and replication based.  

.. toctree::
   :maxdepth: 1
   
  estimation

.. [#w2007] Wolter, K.M. (2007), *Introduction to Variance Estimate, 2nd edn.*, 
   Springer-Verlag New York, Inc

Small Area Estimation (SAE)
---------------------------

SAE techniques can be considered as extensions of the survey estimation techniques that only uses the sample data to produce estimates. Note that we include model assisted techniques such as the Generalized Regression (GREG) estimator in this category. These are often referred to as the direct estimates. In some applications, sample sizes are not large enough to produce reliable direct estimates for the domains of interest. In these situations, one possible way to improve the domain level estimates is to use SAE techniques which require availability of auxiliary information. Depending on the SAE method used, the auxiliary information needs to be available at the domains aggregated level or for non-sampled observations in the target population. There are two main categories of SAE methods that is the area level and the unit level, discussed in this tutorial. 

.. toctree::
   :maxdepth: 1
   
   eblup_area_model
   eblup_unit_model
   eb_unit_model

For a comprehensive review of the small area estimation models and its applications, 
see Rao, J.N.K. and Molina, I. (2015) [#rm2015]_.

.. [#ms2001] McCulloch, C.E.and Searle, S.R. (2001), *Generalized, Linear, Mixed Models*, 
   New York: John Wiley & Sons, Inc.
.. [#rm2015] Rao, J.N.K. and Molina, I. (2015), *Small area estimation, 2nd edn.*, 
   John Wiley & Sons, Hoboken, New Jersey.