from __future__ import annotations

# from typing import Any, Callable, Optional, Union

# import numpy as np
# import pandas as pd
import statsmodels.api as sm


class SurveyGLM:
    """General linear models under complex survey sampling"""

    def __init__(self):
        pass

    def fit(self, y, x):
        glm_model = sm.GLM(y, x)
        glm_results = glm_model.fit()

        return glm_results
