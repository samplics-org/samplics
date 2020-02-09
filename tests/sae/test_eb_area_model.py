import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_area_model import AreaModel

milk = pd.read_csv("./tests/sae/milk.csv")

print(milk)