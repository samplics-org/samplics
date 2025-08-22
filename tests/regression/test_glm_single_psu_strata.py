import numpy as np
import polars as pl
import pytest

from samplics import ModelType
from samplics.regression import SurveyGLM
from samplics.utils.types import SinglePSUEst

sample_data1 = pl.DataFrame(
    {
        "region": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "district": [1, 2, 2, 1, 2, 1, 2, 2, 1, 1],
        "area": [1, 1, 2, 1, 2, 1, 1, 1, 1, 1],
        "domain": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 4.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 98, 98, 98],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    }
)


@pytest.mark.xfail
def test_reg_logistic_single_psu1():
    x = sample_data1.select("age").insert_column(
        0, pl.Series("intercept", np.ones(sample_data1.shape[0]))
    )
    x_cat = sample_data1.select(["area"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=sample_data1["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
    )


def test_reg_logistic_single_psu2():
    x = sample_data1.select("age").insert_column(
        0, pl.Series("intercept", np.ones(sample_data1.shape[0]))
    )
    x_cat = sample_data1.select(["area"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=sample_data1["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.skip,
    )


def test_reg_logistic_single_psu3():
    x = sample_data1.select("age").insert_column(
        0, pl.Series("intercept", np.ones(sample_data1.shape[0]))
    )
    x_cat = sample_data1.select(["area"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=sample_data1["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.certainty,
    )


def test_reg_logistic_single_psu4():
    x = sample_data1.select("age").insert_column(
        0, pl.Series("intercept", np.ones(sample_data1.shape[0]))
    )
    x_cat = sample_data1.select(["area"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=sample_data1["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.combine,
        strata_comb={4: 3},
    )


sample_data2 = pl.DataFrame(
    {
        "region": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
        "district": [2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 5, 5, 5],
        "area": [1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1],
        "domain": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 4.5, 2.5, 2.5, 2.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 28, 98, 23, 34, 45, 18],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    }
)


def test_reg_logistic_single_psu5():
    x = sample_data2.select("age").insert_column(
        0, pl.Series("intercept", np.ones(sample_data2.shape[0]))
    )
    x_cat = sample_data2.select(["area"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=sample_data2["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        single_psu={1: SinglePSUEst.skip, 4: SinglePSUEst.combine, 5: SinglePSUEst.certainty},
        strata_comb={4: 3},
    )
