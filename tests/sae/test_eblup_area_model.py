import numpy as np
import pandas as pd

from samplics.sae.eblup_area_model import EblupAreaModel

milk = pd.read_csv("./tests/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
X = pd.get_dummies(milk["MajorArea"])
X.loc[:, 1] = 1
sigma_e = milk["SD"]

# REML method
fh_model_reml = EblupAreaModel(method="REML")
fh_model_reml.fit(
    yhat=yhat, X=X, area=area, intercept=False, error_std=sigma_e, tol=1e-4,
)
fh_model_reml.predict(X=X, area=area, intercept=False)


def test_fay_herriot_REML_convergence():
    assert fh_model_reml.convergence["achieved"] is True
    assert fh_model_reml.convergence["iterations"] == 3
    assert fh_model_reml.convergence["precision"] <= 1e-4


def test_fay_herriot_REML_goodness():
    assert np.isclose(fh_model_reml.goodness["loglike"], -9.4034611881, atol=1e-4)
    assert np.isclose(fh_model_reml.goodness["AIC"], 30.806922376, atol=1e-4)
    assert np.isclose(fh_model_reml.goodness["BIC"], 41.3741230705, atol=1e-4)


def test_fay_herriot_REML_fixed_effect():
    assert np.isclose(
        fh_model_reml.fixed_effects,
        np.array([0.9681890, 0.1327801, 0.2269462, -0.2413011]),
        atol=1e-4,
    ).all()
    assert np.isclose(
        fh_model_reml.fe_std,
        np.array([0.06936208, 0.10300072, 0.09232981, 0.08161707,]),
        atol=1e-4,
    ).all()


def test_fay_herriot_REML_area_est():
    area_est = np.array(list(fh_model_reml.area_est.values()))
    assert np.isclose(
        area_est,
        np.array(
            [
                1.0219703,
                1.0476018,
                1.0679513,
                0.7608170,
                0.8461574,
                0.9743727,
                1.0584523,
                1.0977762,
                1.2215449,
                1.1951455,
                0.7852155,
                1.2139456,
                1.2096593,
                0.9834967,
                1.1864247,
                1.1556982,
                1.2263411,
                1.2856486,
                1.2363247,
                1.2349600,
                1.0903019,
                1.1923057,
                1.1216470,
                1.2230296,
                1.1938054,
                0.7627195,
                0.7649550,
                0.7338443,
                0.7699294,
                0.6134418,
                0.7695558,
                0.7958250,
                0.7723187,
                0.6102302,
                0.7001782,
                0.7592787,
                0.5298867,
                0.7434466,
                0.7548996,
                0.7701918,
                0.7481164,
                0.8040773,
                0.6810870,
            ]
        ),
        atol=1e-4,
    ).all()


def test_fay_herriot_REML_mse():
    area_mse = np.array(list(fh_model_reml.area_mse.values()))
    assert np.isclose(
        area_mse,
        np.array(
            [
                0.013460220,
                0.005372876,
                0.005701990,
                0.008541740,
                0.009579594,
                0.011670632,
                0.015926137,
                0.010586518,
                0.014184043,
                0.014901472,
                0.007694262,
                0.016336469,
                0.012562726,
                0.012117378,
                0.012031229,
                0.011709147,
                0.010859780,
                0.013690860,
                0.011034674,
                0.013079686,
                0.009948636,
                0.017243977,
                0.011292325,
                0.013625297,
                0.008065787,
                0.009205133,
                0.009205133,
                0.016476912,
                0.007800626,
                0.006098668,
                0.015441564,
                0.014657866,
                0.009024699,
                0.003870786,
                0.007800626,
                0.009646139,
                0.006404335,
                0.010155645,
                0.007209937,
                0.008470277,
                0.005484860,
                0.009205133,
                0.009903626,
            ]
        ),
        atol=1e-4,
    ).all()


# ML method
X = pd.get_dummies(milk["MajorArea"], drop_first=False)
X = np.delete(X.to_numpy(), 0, axis=1)
fh_model_ml = EblupAreaModel(method="ML")
fh_model_ml.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, tol=1e-4,
)
fh_model_ml.predict(
    X=X, area=area,
)


def test_fay_herriot_ML_convergence():
    assert fh_model_ml.convergence["achieved"] is True
    assert fh_model_ml.convergence["iterations"] == 3
    assert fh_model_ml.convergence["precision"] <= 1e-4


def test_fay_herriot_ML_goodness():
    assert np.isclose(fh_model_ml.goodness["loglike"], 1.8172151644, atol=1e-4)
    assert np.isclose(fh_model_ml.goodness["AIC"], 8.3655696710, atol=1e-4)
    assert np.isclose(fh_model_ml.goodness["BIC"], 18.9327703651, atol=1e-4)


def test_fay_herriot_ML_fixed_effect():
    assert np.isclose(
        fh_model_ml.fixed_effects,
        np.array([0.9677986, 0.1278756, 0.2266909, -0.2425804]),
        atol=1e-4,
    ).all()
    assert np.isclose(
        fh_model_ml.fe_std, np.array([0.06590747, 0.09840939, 0.08813973, 0.07753875,]), atol=1e-4,
    ).all()


def test_fay_herriot_ML_area_est():
    area_est = np.array(list(fh_model_ml.area_est.values()))
    assert np.isclose(
        area_est,
        np.array(
            [
                1.0161733,
                1.0436968,
                1.0628168,
                0.7753489,
                0.8554903,
                0.9735857,
                1.0474786,
                1.0953436,
                1.2054092,
                1.1812565,
                0.8033701,
                1.1967756,
                1.1961592,
                0.9914053,
                1.1868829,
                1.1590363,
                1.2232369,
                1.2755188,
                1.2322850,
                1.2304422,
                1.0985768,
                1.1921598,
                1.1279920,
                1.2196285,
                1.1936256,
                0.7590651,
                0.7611232,
                0.7315647,
                0.7662730,
                0.6191454,
                0.7629389,
                0.7863747,
                0.7679782,
                0.6141348,
                0.7013109,
                0.7557564,
                0.5406643,
                0.7411316,
                0.7524506,
                0.7662423,
                0.7465360,
                0.7971405,
                0.6840976,
            ]
        ),
        atol=1e-4,
    ).all()


def test_fay_herriot_ML_mse():
    area_mse = np.array(list(fh_model_ml.area_mse.values()))
    assert np.isclose(
        area_mse,
        np.array(
            [
                0.013579953,
                0.005512868,
                0.005850584,
                0.008735454,
                0.009774528,
                0.011840745,
                0.015934511,
                0.010821811,
                0.014345961,
                0.015036089,
                0.007911095,
                0.016404523,
                0.012771021,
                0.012334611,
                0.012192499,
                0.011877067,
                0.011041285,
                0.013805155,
                0.011213853,
                0.013213712,
                0.010138289,
                0.017193729,
                0.011467631,
                0.013741841,
                0.008251436,
                0.009344874,
                0.009344874,
                0.016390149,
                0.007941642,
                0.006222263,
                0.015404327,
                0.014655741,
                0.009165440,
                0.003946977,
                0.007941642,
                0.009782385,
                0.006532467,
                0.010285994,
                0.007347143,
                0.008612538,
                0.005597690,
                0.009344874,
                0.010037140,
            ]
        ),
        atol=1e-4,
    ).all()


# FH method
X = pd.get_dummies(milk["MajorArea"], drop_first=True)
# X = np.delete(X.to_numpy(), 0, axis=1)
fh_model_fh = EblupAreaModel(method="FH")
fh_model_fh.fit(
    yhat=yhat, X=X, area=area, intercept=True, error_std=sigma_e, tol=1e-4,
)
fh_model_fh.predict(X=X, area=area, intercept=True)


def test_fay_herriot_FH_convergence():
    assert fh_model_fh.convergence["achieved"] is True
    assert fh_model_fh.convergence["iterations"] == 5
    assert fh_model_fh.convergence["precision"] <= 1e-4


def test_fay_herriot_FH_goodness():
    assert np.isclose(fh_model_fh.goodness["loglike"], 1.76370010, atol=1e-4)
    assert np.isclose(fh_model_fh.goodness["AIC"], 8.472599789, atol=1e-4)
    assert np.isclose(fh_model_fh.goodness["BIC"], 19.039800483, atol=1e-4)  # a


def test_fay_herriot_FH_area_est():
    area_est = np.array(list(fh_model_fh.area_est.values()))
    assert np.isclose(
        area_est,
        np.array(
            [
                1.017975,
                1.044963,
                1.064480,
                0.770692,
                0.852512,
                0.973826,
                1.050856,
                1.096165,
                1.210505,
                1.185640,
                0.797568,
                1.202149,
                1.200458,
                0.988971,
                1.186745,
                1.157992,
                1.224223,
                1.278680,
                1.233565,
                1.231860,
                1.095955,
                1.192212,
                1.125997,
                1.220694,
                1.193687,
                0.760243,
                0.762358,
                0.732288,
                0.767459,
                0.617310,
                0.764996,
                0.789314,
                0.769376,
                0.612861,
                0.700961,
                0.756890,
                0.537193,
                0.741881,
                0.753251,
                0.767518,
                0.747059,
                0.799363,
                0.683160,
            ]
        ),
        atol=1e-4,
    ).all()


def test_fay_herriot_FH_mse():
    area_mse = np.array(list(fh_model_fh.area_mse.values()))
    assert np.isclose(
        area_mse,
        np.array(
            [
                0.012757016,
                0.005314467,
                0.005632201,
                0.008323471,
                0.009283520,
                0.011178151,
                0.014867662,
                0.010252709,
                0.013470878,
                0.014094867,
                0.007558331,
                0.015325274,
                0.012038944,
                0.011640323,
                0.011466958,
                0.011181888,
                0.010423673,
                0.012914353,
                0.010580561,
                0.012385544,
                0.009599980,
                0.015890240,
                0.010810966,
                0.012857861,
                0.007864537,
                0.008855177,
                0.008855177,
                0.015041526,
                0.007569376,
                0.005975211,
                0.014211800,
                0.013571949,
                0.008691547,
                0.003833361,
                0.007569376,
                0.009253153,
                0.006264329,
                0.009709450,
                0.007020470,
                0.008185874,
                0.005391055,
                0.008855177,
                0.009484220,
            ]
        ),
        atol=1e-4,
    ).all()
