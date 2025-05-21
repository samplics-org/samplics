import numpy as np
import polars as pl

from samplics import ModelType
from samplics.regression import SurveyGLM

data_str = """
P  F  68   1  No   B  M  74  16  No  P  F  67  30  No
P  M  66  26  Yes  B  F  67  28  No  B  F  77  16  No
A  F  71  12  No   B  F  72  50  No  B  F  76   9  Yes
A  M  71  17  Yes  A  F  63  27  No  A  F  69  18  Yes
B  F  66  12  No   A  M  62  42  No  P  F  64   1  Yes
A  F  64  17  No   P  M  74   4  No  A  F  72  25  No
P  M  70   1  Yes  B  M  66  19  No  B  M  59  29  No
A  F  64  30  No   A  M  70  28  No  A  M  69   1  No
B  F  78   1  No   P  M  83   1  Yes B  F  69  42  No
B  M  75  30  Yes  P  M  77  29  Yes P  F  79  20  Yes
A  M  70  12  No   A  F  69  12  No  B  F  65  14  No
B  M  70   1  No   B  M  67  23  No  A  M  76  25  Yes
P  M  78  12  Yes  B  M  77   1  Yes B  F  69  24  No
P  M  66   4  Yes  P  F  65  29  No  P  M  60  26  Yes
A  M  78  15  Yes  B  M  75  21  Yes A  F  67  11  No
P  F  72  27  No   P  F  70  13  Yes A  M  75   6  Yes
B  F  65   7  No   P  F  68  27  Yes P  M  68  11  Yes
P  M  67  17  Yes  B  M  70  22  No  A  M  65  15  No
P  F  67   1  Yes  A  M  67  10  No  P  F  72  11  Yes
A  F  74   1  No   B  M  80  21  Yes A  F  69   3  No
"""

data_lines = data_str.strip().split("\n")
data_values = []
for line in data_lines:
    values = line.split()
    for i in range(0, len(values), 5):
        data_values.append(values[i : i + 5])

neuralgia = (
    pl.DataFrame(
        data_values,
        schema=[
            ("Treatment", pl.String),
            ("Sex", pl.String),
            ("Age", pl.Int32),
            ("Duration", pl.Int32),
            ("Pain", pl.String),
        ],
        orient="row",
    )
    .rename(mapping=str.lower)
    .with_columns(
        pl.when(pl.col("pain") == "Yes")
        .then(pl.lit(0.0))
        .otherwise(pl.lit(1.0))
        .alias("y")
    )
)


# Missing data

# def test_reg_logistic_missing():
#     y_bin[20] = np.nan
#     y_bin[35] = np.nan
#     x[33, 1] = np.nan
#     weight[55] = np.nan

#     svyglm = SurveyGLM(model=ModelType.LOGISTIC)
#     svyglm.estimate(y=y_bin, x=x, samp_weight=weight, remove_nan=True)

## Categorical data


def test_reg_logistic_categorical_factors():
    x = neuralgia.select("age").insert_column(
        0, pl.Series("intercept", np.ones(neuralgia.shape[0]))
    )
    x_cat = neuralgia.select(["sex", "treatment"])
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=neuralgia["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
    )


api_strat = pl.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
y_bin = (api_strat["api00"].to_numpy() > 743).astype(
    float
)  # api_strat["api00"].quantile(0.75)
x = api_strat.select(["ell", "meals", "mobility"])
x.insert_column(0, pl.Series("intercept", np.ones(x.shape[0])))
stratum = api_strat["stype"]
psu = api_strat["dnum"]
weight = api_strat["pw"]

## Logistic regression


def test_reg_logistic_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.4718306, 0.03363393, 0.01460186, 0.01478849]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.74098503, -0.10295378, -0.113307, -0.0334135]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.590527, 0.02888879, -0.05606876, 0.02455631]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.70295826, 0.90216867, 0.89287651, 0.96713857]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [36.25317644, 1.02931011, 0.94547412, 1.0248603]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4718306**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0336339**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0146019**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0147885**2)


def test_reg_logistic_psu_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight, psu=psu)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.49495812, 0.0318235, 0.0143854, 0.01537275]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.69565592, -0.09940542, -0.11288275, -0.03455863]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.63585611, 0.02534042, -0.056493, 0.02570144]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.45021971, 0.90537558, 0.89325539, 0.9660317]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [37.9343151, 1.02566422, 0.9450731, 1.02603457]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4949581**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0318235**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0143854**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0153727**2)


def test_reg_logistic_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight, stratum=stratum)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.43944329, 0.0336371, 0.01433075, 0.01483659]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.80446299, -0.10296, -0.11277563, -0.03350777]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.52704904, 0.028895, -0.05660012, 0.02465058]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [6.07670735, 0.90216306, 0.89335108, 0.9670474]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [34.02341764, 1.02931651, 0.94497187, 1.02495692]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4394433**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0336371**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0143307**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0148366**2)


def test_reg_logistic_psu_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=y_bin,
        x=x,
        samp_weight=weight,
        psu=psu,
        stratum=stratum,
    )

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.47239534, 0.03241909, 0.01428377, 0.015463]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.73987817, -0.10057275, -0.11268356, -0.03473552]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.59163386, 0.02650775, -0.0566922, 0.02587833]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.69664938, 0.90431932, 0.89343334, 0.96586084]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [36.29332578, 1.02686221, 0.94488486, 1.02621608]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4723953**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0324191**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0142838**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.015463**2)


## Linear regression


def test_reg_linear_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LINEAR)
    svyglm.estimate(y=y, x=x, samp_weight=weight)

    assert np.isclose(
        svyglm.beta["point_est"],
        [8.20887316e02, -4.80586612e-01, -3.14153531e00, 2.25713210e-01],
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [10.97090909, 0.39717553, 0.29173325, 0.4012498]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"],
        [7.99384729e02, -1.25903634e00, -3.71332197e00, -5.60721940e-01],
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"],
        [8.42389903e02, 2.97863116e-01, -2.56974865e00, 1.01214836e00],
    ).all()

    assert svyglm.odds_ratio == {}

    assert np.isclose(svyglm.beta_cov[0, 0], 10.97091**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.3971755**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.2917333**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.4012498**2)


# def test_reg_linear_psu_not_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, psu=psu)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 11.65382**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.4292165**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.285711**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.4159033**2)


# def test_reg_linear_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, stratum=stratum)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 10.25649**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.3977075**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.2883001**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.4026908**2)


# def test_reg_linear_psu_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, psu=psu, stratum=stratum)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 10.10296**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.4109377**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.2743919**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.3873394**2)

## Deterministic and user designated reference

def test_logistic_deterministic_reference():
    """Test that dummy creation is deterministic when no reference is specified."""
    df = pl.DataFrame({
        "cat": ["B", "A", "C", "B", "C", "A"],
        "y": [1, 0, 1, 0, 1, 0],
        "x": [10, 20, 30, 40, 50, 60]
    })

    x = df.select("x")
    x.insert_column(0, pl.Series("intercept", np.ones(x.shape[0])))
    x_cat = df.select("cat")

    svyglm_1 = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_1.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=["cat"]
    )
    labels_1 = svyglm_1.x_labels.copy()

    svyglm_2 = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_2.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=["cat"]
    )
    labels_2 = svyglm_2.x_labels.copy()

    svyglm_3 = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_3.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=["cat"]
    )
    labels_3 = svyglm_3.x_labels.copy()

    assert labels_1 == labels_2
    assert labels_1 == labels_3


def test_logistic_user_specified_reference():
    """Test that specifying reference category produces correct dummies."""
    df = pl.DataFrame({
        "cat": ["B", "A", "C", "B", "C", "A"],
        "y": [1, 0, 1, 0, 1, 0],
        "x": [10, 20, 30, 40, 50, 60]
    })

    x = df.select("x")
    x.insert_column(0, pl.Series("intercept", np.ones(x.shape[0])))
    x_cat = df.select("cat")

    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=["cat"],
        x_cat_reference={"cat": "B"},
    )
    assert any("cat_A" in s for s in svyglm.x_labels)
    assert any("cat_C" in s for s in svyglm.x_labels)
    assert not any("cat_B" in s for s in svyglm.x_labels), "Reference B should not appear"

    svyglm_alt = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_alt.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=["cat"],
        x_cat_reference={"cat": "C"},
    )
    assert not any("cat_C" in s for s in svyglm_alt.x_labels), "Reference C should not appear"
    assert any("cat_A" in s for s in svyglm_alt.x_labels)
    assert any("cat_B" in s for s in svyglm_alt.x_labels)

def test_logistic_two_categorical_predictors():
    """Test dummy encoding for two categorical variables with and without reference."""
    df = pl.DataFrame({
        "cat1": ["B", "A", "C", "B", "C", "A"],
        "cat2": ["X", "Y", "Z", "X", "Z", "Y"],
        "y": [1, 0, 1, 0, 1, 0],
        "x": [10, 20, 30, 40, 50, 60]
    })

    x = df.select("x")
    x.insert_column(0, pl.Series("intercept", np.ones(x.shape[0])))
    x_cat = df.select(["cat1", "cat2"])

    # Without specifying reference
    svyglm_default = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_default.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
    )
    labels_default = svyglm_default.x_labels.copy()

    # With user-specified reference
    svyglm_custom = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_custom.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
        x_cat_reference={"cat1": "C", "cat2": "Z"},
    )
    labels_custom = svyglm_custom.x_labels.copy()

    # Assert both contain expected dummy variables
    assert any("cat1_A" in s for s in labels_custom)
    assert any("cat1_B" in s for s in labels_custom)
    assert not any("cat1_C" in s for s in labels_custom)

    assert any("cat2_X" in s for s in labels_custom)
    assert any("cat2_Y" in s for s in labels_custom)
    assert not any("cat2_Z" in s for s in labels_custom)

    # Consistency check for default reference behavior
    svyglm_repeat = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm_repeat.estimate(
        y=df["y"].to_numpy(),
        x=x.to_numpy(),
        x_labels=x.columns,
        x_cat=x_cat,
        x_cat_labels=x_cat.columns,
    )
    assert svyglm_repeat.x_labels == labels_default, "Default dummy encoding is not deterministic"
