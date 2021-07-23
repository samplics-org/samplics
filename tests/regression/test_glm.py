import numpy as np
import pandas as pd

from samplics.regression import SurveyGLM


api_strat = pd.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
x = api_strat[["ell", "meals", "mobility"]]
x.insert(0, "intercept", 1)

svyglm = SurveyGLM()
svyglm_results = svyglm.fit(y=y, x=x)

svyglm_results.summary()

breakpoint()

