
Tutorial
========

Samplics is a Python package designed to be a comprehensive tool for selecting, weighting and analyzing survey sample data obtained from complex designs. The main objective of this tutorial is to take the user through the Samplics' APIs main features. We hope that after going through the tutorial, the user will have a good understanding of APIs' use cases. The tutorial is organized in several main topics. The user should be able to only focus on topics of interest. 

This tutorial is intended for users with basic knowledge of Python. It is assumed that the user has a basic understanding of Python syntax. Also, the tutorial is not intended to teach survey sampling methods. To learn survey sampling methods, we refer the user to the `UNStats Handbook 2005 <https://unstats.un.org/unsd/hhsurveys/>`_, `Designing Household Survey Samples: Practical Guidelines <https://unstats.un.org/unsd/demographic/sources/surveys/Series_F98en.pdf>`_, and the reference material mentioned throughout this tutorial. 


Sample size calculation
-----------------------
Sample size calculation and allocation is usually one of the first steps in conducting a survey. Sample size calculations allow the survey implementers to collect enough data o achieve targeted precision levels. To do so, the sample size calculation should follow as close as possible the actual sampling design and take into account imperfections of survey implementations such as non response. Since sample sizes are calculated prior to implementing the survey, many of the input data needed to calculate sample sizes are obtained from previous experience and from reasonable assumptions. 

In this tutorial, we illustrate the calculation and allocation of some common sample size methods. Users can found additional details from Chow, S., Shao, J., Wang, H., Lokhnygina, Y. (2018) [#chow2018]_ and Ryan, T.P. [#ryan2013]_.

.. toctree::
   :maxdepth: 1

   sample_size_calculation


.. [#chow2018] Chow, S., Shao, J., Wang, H., Lokhnygina, Y. (2018), *Sample Size Calculations in Clinical Research, Third Edition.*, CRC Press, Taylor & Francis Group.
.. [#ryan2013] Ryan, T.P. (2013), *Sample Size Determination and Power*, Jonh Wiley & Sons, Inc.

Sample selection
----------------

When a sample frame is available, the simple random sampling (srs) method is usually the easiest way to select a sample. However, operational and cost constraints often result in much more complex mechanisms for selecting samples. Common use cases of complex sampling designs are the national household surveys such as the `Demographic and Health Surveys (DHS) <https://dhsprogram.com>`_, the `Living Standards Measurement Study (LSMS) <http://surveys.worldbank.org>`_, the `Multiple Indicator Cluster Surveys (MICS) <https://mics.unicef.org>`_, and many more. In the national household surveys, the sample is selected in stages. For example, at the first stage clusters of households are selected using probability proportional to size (pps) method, at the second stage households are selected from the sampled clusters, and at the final stage individuals are selected from households in the sample.  

In this tutorial, we generate simulated data to illustrate the first and second stage selections of clusters and households, respectively. 

.. toctree::
   :maxdepth: 1
   
   psu_selection
   ssu_selection

For a comprehensive review of complex sampling techniques, users may want to consult Brewer, K.R.W. and Hanif, M. (1983) [#brewer1983]_, Cochran, W.G. (1977) [#cochran1977]_, Kish, L. (1965) [#kish1965]_, and Lohr, S.L. (2010) [#lohr2010]_.

.. [#brewer1983] Brewer, K.R.W. and Hanif, M. (1983), *Sampling With Unequal Probabilities*,  
   Springer-Verlag New York, Inc
.. [#cochran1977] Cochran, W.G. (1977), *Sampling Techniques, 3rd edn.*, Jonh Wiley & Sons, Inc.
.. [#kish1965] Kish, L. (1965), *Survey Sampling*, Jonh Wiley & Sons, Inc.
.. [#lohr2010] Lohr, S.L. (2010), *Sampling: Design and Analysis, 2nd edn.*, Cengage Learning, Inc.

Sample weighting
----------------

The sample weighting is the mechanism that allows us to generalize the results from the sample to the target population. The design/base weights are obtained from the final probability of selection as their inverse. In large scale surveys, design weights are adjusted to account for non-response, for extreme values, or to ensure that auxiliary variables benchmark to known controls. 

The weighting tutorial is in two parts. The first part discusses several weight adjustments methods e.g. non-response adjustment, poststratification, calibration. The second part of the tutorial walks the user through the creation of replicate weights using Booststrap, Balanced Repeated Replication (BRR), and Jackknife methods. 

.. toctree::
   :maxdepth: 1
   
   sample_weights
   replicate_weights

Valliant, R. and Dever, J. A. (2018) [#valliant2018]_ provides a step-by-step guide on calculating 
sample weights.

.. [#valliant2018] Valliant, R. and Dever, J. A. (2018), *Survey Weights: A Step-by-Step Guide to       
   Calculation*, Stata Press.

Population parameters estimation
--------------------------------

Often the goal of conducting a survey is to estimate parameters of the target population from the survey sample. To appropriately estimate uncertainties associated with the sample estimates, the complex design must be taken into account. In this tutorial, we demonstrate how to use the APIs to produce point estimates of mean, total, proportion and ratio parameters as well as their associated Taylor-based and replication-based measures of uncertainty. Wolter, K.M. (2007) [#wolter2007]_ provides a self-contained description of a number of techniques for variance estimation both Taylor and replication based.  

.. toctree::
   :maxdepth: 1
   
   estimation

.. [#wolter2007] Wolter, K.M. (2007), *Introduction to Variance Estimate, 2nd edn.*, 
   Springer-Verlag New York, Inc
   
Categorical data analysis
-------------------------

Categorical data analysis refers to the set of methods for analyzing categorical response variables. This is a large topic in data analysis and used in the context of complex survey sampling. Contingency tables or cross-tabulations with associated statistics are well-known categorical data analysis techniques. But categorical data analysis extends to more areas such as modeling techniques for categorical response variables e.g. logistic/logit/loglinear regressions and generalized linear mixed models for categorical responses. Agresti, Alan (2013) [#agresti2013]_ provides an overview of these methods. 

.. toctree::
   :maxdepth: 1
   
   tabulation
   ttest

.. [#agresti2013] Agresti, Alan (2013), *Categorical Data Analysis, 3rd edn.*, 
    Hoboken, NJ and Chichester: Wiley; John Wiley [distributor]

Small Area Estimation (SAE)
---------------------------

SAE techniques can be considered as extensions of the survey estimation techniques that only use the sample data to produce estimates. Note that we include model assisted techniques such as the Generalized Regression (GREG) estimator in this category. These are often referred to as the direct estimates. In some applications, sample sizes are not large enough to produce reliable direct estimates for the domains of interest. In these situations, one possible way to improve the domain level estimates is to use SAE techniques which require availability of auxiliary information. Depending on the SAE method used, the auxiliary information needs to be available at the domains aggregated level or for non-sampled observations in the target population. SAE techniques produce modeled estimates of parameters of interest. There are two main categories of SAE methods that are the area level and the unit level, discussed in this tutorial. 

.. toctree::
   :maxdepth: 1
   
   eblup_area_model
   eblup_unit_model

Generalized linear mixed model are the statistical framework used to develop the SAE methods, for an introduction to GLMM see McCulloch, C.E.and Searle, S.R. (2001) [#mcculloch2001]_. For a comprehensive review of the small area estimation models and its applications, see Rao, J.N.K. and Molina, I. (2015) [#rao2015]_. 

.. [#mcculloch2001] McCulloch, C.E.and Searle, S.R. (2001), *Generalized, Linear, Mixed Models*, 
   New York: John Wiley & Sons, Inc.
.. [#rao2015] Rao, J.N.K. and Molina, I. (2015), *Small area estimation, 2nd edn.*, 
   John Wiley & Sons, Hoboken, New Jersey.