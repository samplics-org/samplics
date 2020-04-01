import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

import sim_pop

eblup_bhf_reml = UnitModel()
eblup_bhf_reml.fit(y_s, X_s, area_s)

# eblup_bhf_reml.predict(X_s, X_smean, area_s)

# print(eblup_bhf_reml.gamma, "\n")
