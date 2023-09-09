import numpy as np
import pandas as pd

from samplics.sae.eblup_unit_model import EblupUnitModel


cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

cornsoybean = cornsoybean.sample(frac=1)  # shuffle the data to remove the
# print(cornsoybean)

areas = cornsoybean["County"]
areas_list = np.unique(areas)

ys = cornsoybean["CornHec"]
Xs = cornsoybean[["CornPix", "SoyBeansPix"]]

Xmean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]
# print(Xmean)

samp_size = np.array([1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6])
pop_size = np.array([545, 566, 394, 424, 564, 570, 402, 567, 687, 569, 965, 556])

"""REML Method"""
eblup_bhf_reml = EblupUnitModel()
eblup_bhf_reml.fit(
    ys,
    Xs,
    areas,
)

eblup_bhf_reml.predict(Xmean, areas_list)


def test_eblup_bhf_reml():
    assert eblup_bhf_reml.method == "REML"


def test_fixed_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.fixed_effects,
        np.array([17.96398, 0.3663352, -0.0303638]),
        atol=1e-6,
    ).all()


def test_fe_std_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.fe_std,
        np.array([30.986801, 0.065101, 0.067583]),
        atol=1e-6,
    ).all()


def test_gamma_bhf_reml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.gamma.values())),
        np.array(
            [
                0.17537405,
                0.17537405,
                0.17537405,
                0.29841402,
                0.38950426,
                0.38950426,
                0.38950426,
                0.38950426,
                0.45965927,
                0.51535245,
                0.51535245,
                0.56063774,
            ]
        ),
        atol=1e-6,
    ).all()


def test_random_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.random_effects,
        np.array(
            [
                2.184574,
                1.475118,
                -4.730863,
                -2.764825,
                8.370915,
                4.274827,
                -2.705540,
                1.156682,
                5.026852,
                -2.883398,
                -8.652532,
                -0.751808,
            ]
        ),
        atol=1e-6,
    ).all()


def test_re_std_bhf_reml():
    assert np.isclose(eblup_bhf_reml.re_std**2, 63.3149, atol=1e-6)


def test_error_var_bhf_reml():
    assert np.isclose(eblup_bhf_reml.error_std**2, 297.7128, atol=1e-6)


def test_goodness_of_fit_bhf_reml():
    assert np.isclose(eblup_bhf_reml.goodness["loglike"], -161.005759)
    assert np.isclose(eblup_bhf_reml.goodness["AIC"], 326.011518)
    assert np.isclose(eblup_bhf_reml.goodness["BIC"], 329.064239)


def test_convergence_bhf_reml():
    assert eblup_bhf_reml.convergence["achieved"] == True
    assert eblup_bhf_reml.convergence["iterations"] == 4


def test_area_estimate_bhf_reml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.area_est.values())),
        np.array(
            [
                122.56367092,
                123.51515946,
                113.09071900,
                115.02074400,
                137.19621212,
                108.94543201,
                116.51553231,
                122.76148230,
                111.53048000,
                124.18034553,
                112.50472697,
                131.25788283,
            ]
        ),
        atol=1e-6,
    ).all()


# @pytest.mark.skip(reason="to be fixed")
def test_area_mse_bhf_reml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.area_mse.values())),
        np.array(
            [
                85.495399459,
                85.648949504,
                85.004705566,
                83.235995880,
                72.017014455,
                73.356967955,
                72.007536645,
                73.580035237,
                65.299062174,
                58.426265442,
                57.518251822,
                53.876770532,
            ]
        ),
        atol=1e-6,
    ).all()


eblup_bhf_reml_fpc = EblupUnitModel()
eblup_bhf_reml_fpc.fit(ys, Xs, areas)

eblup_bhf_reml_fpc.predict(Xmean, areas_list, pop_size)


def test_y_predicted_bhf_reml_fpc():
    assert np.isclose(
        np.array(list(eblup_bhf_reml_fpc.area_est.values())),
        np.array(
            [
                122.582519,
                123.527414,
                113.034260,
                114.990082,
                137.266001,
                108.980696,
                116.483886,
                122.771075,
                111.564754,
                124.156518,
                112.462566,
                131.251525,
            ]
        ),
        atol=1e-6,
    ).all()


def test_bhf_reml_to_dataframe_default():
    df = eblup_bhf_reml.to_dataframe()
    assert df.shape[1] == 4
    assert (df.columns == ["_parameter", "_area", "_estimate", "_mse"]).all()


def test_bhf_reml_to_dataframe_not_default():
    df = eblup_bhf_reml.to_dataframe(
        col_names=["parameter", "small_area", "modelled_estimate", "taylor_mse"]
    )
    assert df.shape[1] == 4
    assert (
        df.columns == ["parameter", "small_area", "modelled_estimate", "taylor_mse"]
    ).all()


# Bootstrap with REML
eblup_bhf_reml_boot = EblupUnitModel()
eblup_bhf_reml_boot.fit(
    ys,
    Xs,
    areas,
)
eblup_bhf_reml_boot.predict(Xmean, areas_list)
eblup_bhf_reml_boot.bootstrap_mse(number_reps=5, show_progress=False)

df1_reml = eblup_bhf_reml_boot.to_dataframe()


def test_bhf_reml_to_dataframe_boot_default():
    assert df1_reml.shape[1] == 5
    assert (
        df1_reml.columns == ["_parameter", "_area", "_estimate", "_mse", "_mse_boot"]
    ).all()


df2_reml = eblup_bhf_reml_boot.to_dataframe(
    col_names=["parameter", "small_area", "modelled_estimate", "taylor_mse", "boot_mse"]
)


def test_bhf_reml_to_dataframe_boot_not_default():
    assert df2_reml.shape[1] == 5
    assert (
        df2_reml.columns
        == ["parameter", "small_area", "modelled_estimate", "taylor_mse", "boot_mse"]
    ).all()


# Shorter output
np.random.seed(123)

samp_size_short = np.array([3, 3, 3, 4, 5, 5, 6])
pop_size_short = np.array([570, 402, 567, 687, 569, 965, 556])
pop_area_short = np.linspace(6, 12, 7).astype(int)
Xp_mean_short = Xmean.loc[5:12, :]

eblup_bhf_reml_short = EblupUnitModel()
eblup_bhf_reml_short.fit(ys, Xs, areas, intercept=True)
eblup_bhf_reml_short.predict(Xp_mean_short, pop_area_short, pop_size_short)


def test_area_estimate_bhf_reml_short():
    assert np.isclose(
        np.array(list(eblup_bhf_reml_short.area_est.values())),
        np.array(
            [
                108.98069631,
                116.48388625,
                122.77107460,
                111.56475375,
                124.15651773,
                112.46256629,
                131.25152478,
            ]
        ),
        atol=1e-6,
    ).all()


# @pytest.mark.skip(reason="to be fixed")
def test_area_mse_bhf_reml_short():
    assert np.isclose(
        np.array(list(eblup_bhf_reml_short.area_mse.values())),
        np.array(
            [
                78.70883983,
                78.02323786,
                78.87309307,
                70.04040931,
                64.11261351,
                61.87654547,
                59.81982861,
            ]
        ),
        atol=1e-6,
    ).all()


"""ML Method"""
eblup_bhf_ml = EblupUnitModel(method="ml")
eblup_bhf_ml.fit(ys, Xs, areas)

eblup_bhf_ml.predict(Xmean, areas_list)


def test_eblup_bhf_ml():
    assert eblup_bhf_ml.method == "ML"


def test_fixed_effects_bhf_ml():
    assert np.isclose(
        eblup_bhf_ml.fixed_effects,
        np.array([18.08888, 0.36566, -0.03017]),
        atol=1e-5,
    ).all()


def test_fe_std_bhf_ml():
    assert np.isclose(
        eblup_bhf_ml.fe_std,
        np.array([29.82724469, 0.06262676, 0.06506189]),
        atol=1e-5,
    ).all()


def test_gamma_bhf_ml():
    assert np.isclose(
        np.array(list(eblup_bhf_ml.gamma.values())),
        np.array(
            [
                0.14570573,
                0.14570573,
                0.14570573,
                0.25435106,
                0.33848019,
                0.33848019,
                0.33848019,
                0.33848019,
                0.40555003,
                0.46027174,
                0.46027174,
                0.50576795,
            ]
        ),
        atol=1e-6,
    ).all()


def test_random_effects_bhf_ml():
    assert np.isclose(
        eblup_bhf_ml.random_effects,
        np.array(
            [
                1.8322323,
                1.2218437,
                -3.9308431,
                -2.3261989,
                7.2988558,
                3.7065346,
                -2.3371090,
                1.0315879,
                4.4367420,
                -2.5647926,
                -7.7046350,
                -0.6642178,
            ]
        ),
        atol=1e-6,
    ).all()


def test_re_std_bhf_ml():
    assert np.isclose(eblup_bhf_ml.re_std**2, 47.79559, atol=1e-4)


def test_error_var_bhf_ml():
    assert np.isclose(eblup_bhf_ml.error_std**2, 280.2311, atol=1e-4)


def test_goodness_of_fit_bhf_ml():
    assert np.isclose(eblup_bhf_ml.goodness["loglike"], -159.1981)
    assert np.isclose(eblup_bhf_ml.goodness["AIC"], 328.4, atol=0.1)
    assert np.isclose(eblup_bhf_ml.goodness["BIC"], 336.5, atol=0.1)


def test_convergence_bhf_ml():
    assert eblup_bhf_ml.convergence["achieved"] == True
    assert eblup_bhf_ml.convergence["iterations"] == 3


def test_area_estimate_bhf_ml():
    assert np.isclose(
        np.array(list(eblup_bhf_ml.area_est.values())),
        np.array(
            [
                122.17284832,
                123.22129485,
                113.85918468,
                115.42994973,
                136.06978025,
                108.37573030,
                116.84704244,
                122.60003878,
                110.93542654,
                124.44934607,
                113.41480260,
                131.28369873,
            ]
        ),
        atol=1e-6,
    ).all()


def test_area_mse_bhf_ml():
    assert np.isclose(
        np.array(list(eblup_bhf_ml.area_mse.values())),
        np.array(
            [
                70.03789330,
                70.14078955,
                69.75891524,
                71.50874622,
                64.73862949,
                66.13552266,
                64.77099780,
                66.09246929,
                60.71287515,
                55.31330901,
                54.52024143,
                51.85801645,
            ]
        ),
        atol=1e-4,
    ).all()


eblup_bhf_ml_fpc = EblupUnitModel(method="ML")
eblup_bhf_ml_fpc.fit(ys, Xs, areas)

eblup_bhf_ml_fpc.predict(Xmean, areas_list, pop_size)


def test_area_est_bhf_ml_fpc():
    assert np.isclose(
        np.array(list(eblup_bhf_ml_fpc.area_est.values())),
        np.array(
            [
                122.1926,
                123.2340,
                113.8007,
                115.3978,
                136.1457,
                108.4139,
                116.8129,
                122.6107,
                110.9733,
                124.4229,
                113.3680,
                131.2767,
            ]
        ),
        atol=1e-4,
    ).all()


# Bootstrap with ML
eblup_bhf_ml_boot = EblupUnitModel(method="ML")
eblup_bhf_ml_boot.fit(
    ys,
    Xs,
    areas,
)
eblup_bhf_ml_boot.predict(Xmean, areas_list)
eblup_bhf_ml_boot.bootstrap_mse(number_reps=5, show_progress=False)

df1_ml = eblup_bhf_ml_boot.to_dataframe()


def test_bhf_ml_to_dataframe_boot_default():
    assert df1_ml.shape[1] == 5
    assert (
        df1_ml.columns == ["_parameter", "_area", "_estimate", "_mse", "_mse_boot"]
    ).all()


df2_ml = eblup_bhf_ml_boot.to_dataframe(
    col_names=["parameter", "small_area", "modelled_estimate", "taylor_mse", "boot_mse"]
)


def test_bhf_ml_to_dataframe_boot_not_default():
    assert df2_ml.shape[1] == 5
    assert (
        df2_ml.columns
        == ["parameter", "small_area", "modelled_estimate", "taylor_mse", "boot_mse"]
    ).all()


# Shorter output
eblup_bhf_ml_short = EblupUnitModel(method="ML")
eblup_bhf_ml_short.fit(ys, Xs, areas, intercept=True)
eblup_bhf_ml_short.predict(Xp_mean_short, pop_area_short, pop_size_short)


def test_area_estimate_bhf_ml_short():
    assert np.isclose(
        np.array(list(eblup_bhf_ml_short.area_est.values())),
        np.array(
            [
                108.41385641,
                116.81295596,
                122.61070603,
                110.97329145,
                124.42291775,
                113.36799091,
                131.27669442,
            ]
        ),
        atol=1e-6,
    ).all()


# @pytest.mark.skip(reason="to be fixed")
def test_area_mse_bhf_ml_short():
    assert np.isclose(
        np.array(list(eblup_bhf_ml_short.area_mse.values())),
        np.array(
            [
                71.07422316,
                70.52276075,
                71.03548298,
                65.27922762,
                60.93670432,
                58.91938558,
                57.87424555,
            ]
        ),
        atol=1e-6,
    ).all()
