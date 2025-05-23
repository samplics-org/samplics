import numpy as np
import pandas as pd
import pytest

from samplics.sae.eblup_area_model import EblupAreaModel

df = pd.DataFrame(
    {
        "AREA": np.array(
            [
                "98",
                "99",
                "1111",
                "1121",
                "1131",
                "1151",
                "2111",
                "2123",
                "2131",
                "2211",
                "2212",
                "2213",
                "2361",
                "3113",
                "3114",
                "3115",
                "3116",
                "3117",
                "3118",
                "3121",
                "3132",
                "3133",
                "3141",
                "3152",
                "3162",
                "3219",
                "3222",
                "3231",
                "3252",
                "3253",
                "3254",
                "3256",
                "3261",
                "3262",
                "3271",
                "3272",
                "3311",
                "3322",
                "3327",
                "3331",
                "3332",
                "3339",
                "3342",
                "3344",
                "3345",
                "3351",
                "3352",
                "3361",
                "3364",
                "3366",
                "3371",
                "3391",
                "3399",
                "4236",
                "4237",
                "4238",
                "4242",
                "4243",
                "4244",
                "4249",
                "4251",
                "4411",
                "4412",
                "4413",
                "4421",
                "4431",
                "4441",
                "4451",
                "4452",
                "4453",
                "4461",
                "4471",
                "4481",
                "4482",
                "4483",
                "4511",
                "4512",
                "4522",
                "4523",
                "4531",
                "4533",
                "4539",
                "4543",
                "4811",
                "4831",
                "4841",
                "4851",
                "4881",
                "4921",
                "4931",
                "5111",
                "5121",
                "5151",
                "5173",
                "5174",
                "5182",
                "5191",
                "5211",
                "5222",
                "5231",
                "5241",
                "5311",
                "5411",
                "5412",
                "5413",
                "5414",
                "5415",
                "5416",
                "5419",
                "5611",
                "5613",
                "5614",
                "5616",
                "5617",
                "5621",
                "6111",
                "6112",
                "6114",
                "6116",
                "6211",
                "6212",
                "6213",
                "6214",
                "6215",
                "6216",
                "6221",
                "6231",
                "6232",
                "6241",
                "6242",
                "6243",
                "6244",
                "7111",
                "7121",
                "7131",
                "7211",
                "7223",
                "7224",
                "8111",
                "8114",
                "8121",
                "8122",
                "8123",
                "8129",
                "8131",
                "8132",
                "8139",
                "8141",
            ],
            dtype=object,
        ),
        "AWEEKS": np.array(
            [
                45.64662296950698,
                45.04880774962742,
                42.51691349032937,
                48.36852791878172,
                43.5,
                36.15359477124183,
                51.0,
                51.0,
                51.0,
                51.0,
                51.0,
                51.0,
                50.11086309523809,
                47.92446043165467,
                48.1304347826087,
                51.0,
                41.86668537749088,
                49.72776365946633,
                40.84212900315742,
                36.79842931937173,
                47.26062322946176,
                51.0,
                32.3448275862069,
                48.4559051057246,
                51.0,
                51.0,
                45.25949367088607,
                40.9750122488976,
                49.41059602649007,
                51.0,
                41.40072859744991,
                51.0,
                50.68307484828051,
                51.0,
                51.0,
                51.0,
                43.5,
                51.0,
                51.0,
                51.0,
                51.00000000000001,
                51.0,
                51.0,
                42.90476190476191,
                43.5,
                43.5,
                20.0,
                24.96309963099631,
                47.54634146341463,
                51.0,
                51.0,
                35.9064449064449,
                44.14349217638691,
                51.0,
                51.0,
                22.35570469798658,
                41.73640167364017,
                51.0,
                46.69521360759494,
                51.0,
                51.0,
                47.46165966386555,
                51.0,
                15.0125786163522,
                35.19458128078818,
                51.0,
                49.99038461538461,
                46.31392117230925,
                49.52785714285714,
                51.0,
                46.42091152815014,
                47.19563356164384,
                36.79226736566186,
                51.0,
                51.0,
                46.48661567877629,
                51.0,
                47.28813038130382,
                49.4310766721044,
                49.0589430894309,
                49.55474452554745,
                44.67649857278782,
                38.19845857418112,
                51.0,
                51.0,
                17.0,
                20.0,
                36.8516699410609,
                38.87617260787992,
                34.66457601605619,
                51.0,
                51.0,
                33.0,
                51.0,
                51.0,
                51.0,
                48.5,
                48.0607424071991,
                49.47945205479452,
                45.70260223048327,
                48.140625,
                41.10927318295739,
                51.0,
                51.0,
                51.0,
                51.0,
                18.08270676691729,
                51.0,
                51.0,
                45.29659863945578,
                43.73248407643312,
                51.0,
                51.0,
                42.1876559422193,
                51.0,
                43.79799301919721,
                41.82950631458094,
                50.99999999999999,
                32.36516853932584,
                43.12775330396476,
                51.0,
                48.81081081081081,
                50.18541131105398,
                48.93978349120434,
                42.09174892432296,
                50.9071408004607,
                45.5055055055055,
                44.04714784633295,
                45.9925799086758,
                51.0,
                51.0,
                47.22430203045685,
                51.0,
                51.0,
                44.24724061810154,
                44.31297574791999,
                45.8459185754095,
                51.0,
                38.27210884353742,
                35.76094276094276,
                44.4171212757029,
                51.0,
                47.04709364908503,
                51.0,
                43.3342776203966,
                27.80507497116494,
                51.0,
                43.46071428571429,
            ],
            dtype=np.float64,
        ),
        "AWEEKS_SE": np.array(
            [
                1.533268330876919,
                3.543725954724538,
                1.234338033811715,
                2.241009800721792,
                2.8479072007018487,
                4.712903263441091,
                4.331635409837789,
                6.693919415354036,
                6.723206848139723,
                2.9646691517161203,
                3.0179031960207614,
                4.462717041417528,
                0.5944140551704142,
                4.5133590302481466,
                3.8069574445617773,
                2.744152062967039,
                3.254084641464709,
                0.8807037782408489,
                5.702205712237936,
                7.104346716712396,
                2.7489719538140722,
                4.802389689877974,
                0.7639407213186462,
                1.673556145576195,
                2.45601047397543,
                13.77492479019938,
                2.465447755466721,
                4.812554828499073,
                3.3889144115044743,
                8.985083123339798,
                3.977061129364221,
                3.2893696128469863,
                0.2931010001838909,
                5.903472952911488,
                12.017647882143311,
                6.685942520797158,
                4.723733105891465,
                12.973341307990287,
                3.0631979339366913,
                6.526281920212288,
                9.148164703499338,
                9.148164703499335,
                2.5198169478185433,
                1.8749889299450608,
                9.1771247078275,
                4.576776203264572,
                8.326707112581913,
                8.291919721246147,
                4.282020508435999,
                4.630580917382065,
                4.081620328869058,
                2.1130945008462305,
                3.323382881476829,
                2.9846710243462757,
                3.685866086938653,
                1.3670272223173254,
                2.065975060945947,
                7.225317576494092,
                1.490729174774682,
                0.0,
                3.760754491909658,
                3.238368947362995,
                5.069979697252019,
                2.2402608735568794,
                4.858570824561859,
                6.104355389571342,
                1.062933953922035,
                1.567426529100042,
                1.375390171611767,
                6.9106557811674305,
                3.760430897942874,
                2.333767517409578,
                5.235942175505164,
                11.695887687505984,
                21.165165114984855,
                11.433290899380728,
                28.377043642945154,
                1.316699568030435,
                1.236415787495349,
                2.8955072152641153,
                1.193706007126115,
                4.772898158746642,
                1.032265596976645,
                3.943169095568389,
                20.436221051260617,
                1.3700006004514487,
                4.030050699135388,
                1.8388210393026312,
                3.1949363748184747,
                5.801516886416061,
                6.088148414286218,
                11.116607683795644,
                1.7767950330688433,
                3.593339087095153,
                3.2693917728600166,
                5.899208467439564,
                2.1931358324323065,
                2.224512295109605,
                2.710568045960032,
                2.048682280880299,
                2.552234901765515,
                7.239294605804065,
                3.5432317019337654,
                3.5867520000041555,
                3.645177553209268,
                6.699570216223078,
                2.786545636004336,
                6.052098854962594,
                5.207281855691421,
                5.135778303240716,
                3.549443533152181,
                5.570036810197121,
                5.896756586948745,
                1.62606411252963,
                4.495987065927056,
                3.717405621023141,
                3.253600797458164,
                5.770743786174272,
                7.059185591925254,
                4.505616014506121,
                0.0,
                5.406230351198365,
                0.5044440134255587,
                1.247582069143525,
                4.079063253392679,
                0.101888883235952,
                2.381883515842805,
                6.061697515759663,
                3.571725938609288,
                9.468830365152193,
                12.63220105196682,
                2.566107380637993,
                8.091881252063153,
                7.5532408255327335,
                3.454711504477244,
                1.782513964678307,
                0.6943413449912031,
                0.0,
                5.504219237112557,
                2.737037838992123,
                3.468340768262697,
                10.174407830788331,
                3.247191459007047,
                22.393203028479203,
                8.253293099545209,
                3.169530181040777,
                10.720377836174713,
                2.763931787414776,
            ],
            dtype=np.float64,
        ),
        "AWEEKS_acs5": np.array(
            [
                45.29673926727936,
                46.41486359360301,
                39.54597701149425,
                44.55259900990099,
                17.2948717948718,
                39.41354246365723,
                51.0,
                35.38709677419355,
                43.42222222222222,
                51.0,
                51.0,
                49.58490566037736,
                45.70639379347244,
                43.77435064935065,
                40.21689723320158,
                40.07027027027027,
                44.83698415802855,
                48.16728971962617,
                45.74510932105868,
                41.66105769230769,
                47.63076923076922,
                51.0,
                45.93574297188754,
                45.04968684759916,
                51.0,
                45.91752577319588,
                45.52208201892744,
                41.19923002887392,
                47.13586956521739,
                51.0,
                45.77699530516432,
                47.63529411764706,
                45.27050610820244,
                43.26732673267327,
                51.0,
                51.0,
                48.61904761904762,
                51.0,
                51.0,
                19.10714285714286,
                49.32432432432432,
                49.32432432432432,
                46.80985915492958,
                48.32005141388175,
                40.08196721311475,
                49.01282051282051,
                48.9367816091954,
                41.06901840490798,
                49.17293233082707,
                51.0,
                44.1160458452722,
                45.2669245647969,
                44.25804119985544,
                51.0,
                26.2,
                36.86861313868613,
                47.9321608040201,
                44.53205128205128,
                44.13606148732863,
                47.91740412979351,
                51.0,
                49.3063063063063,
                51.0,
                43.86163522012578,
                36.91226575809199,
                48.80232558139536,
                43.73062015503876,
                45.00955281738889,
                37.30034722222222,
                46.14971751412429,
                46.92549261083744,
                45.8546441495778,
                45.48048279404212,
                44.46875,
                39.30952380952381,
                48.18392070484582,
                51.0,
                45.64794326241135,
                47.18755020080322,
                44.19974554707379,
                44.40438489646773,
                43.20114122681883,
                41.73684210526316,
                45.58370044052864,
                51.0,
                30.7843137254902,
                41.80452674897119,
                48.42085235920852,
                45.22970297029703,
                41.24872611464968,
                47.55555555555556,
                32.6,
                46.40454545454546,
                50.99999999999999,
                51.0,
                38.58620689655172,
                43.95212765957447,
                49.59675675675675,
                47.94152542372881,
                49.89259259259259,
                44.05500705218618,
                47.73338670936749,
                46.20104895104895,
                48.67399267399267,
                49.20618556701031,
                50.55188679245283,
                42.69381107491856,
                46.75111111111111,
                36.69153225806452,
                38.6300129366106,
                39.4227240649259,
                42.50396825396825,
                42.76223776223776,
                43.58410769597332,
                37.02767175572519,
                43.77495232040687,
                41.20081300813008,
                50.62903225806452,
                38.88888888888889,
                45.95388601036269,
                48.09979736575481,
                46.9237668161435,
                46.24006908462867,
                49.66452442159382,
                42.92186613726602,
                48.28283238559437,
                44.52133233532934,
                43.85067681895093,
                44.81551634665283,
                43.63076923076923,
                43.20473537604457,
                42.88233177401943,
                37.22896281800391,
                51.0,
                43.4426272066459,
                43.33704842428901,
                44.99401524607703,
                45.17837338262477,
                44.53575076608784,
                43.11706349206349,
                45.20958805107203,
                51.0,
                46.70761953464587,
                37.03915662650602,
                42.87981859410431,
                37.06440071556351,
                41.5531914893617,
                43.56408661025386,
            ],
            dtype=np.float64,
        ),
    }
)


yhat = df.AWEEKS
X = df[["AWEEKS_acs5"]]
area = df.AREA
error_std = df.AWEEKS_SE


@pytest.mark.xfail(
    strict=True, reason="At least one standard error is NOT strickly positive!"
)
def test_standard_errors_non_positive():
    fh_model_reml = EblupAreaModel(method="REML")
    fh_model_reml.fit(
        yhat=yhat,
        X=X,
        area=area,
        error_std=error_std,
        intercept=True,
        tol=1e-8,
    )
    fh_model_reml.predict(X=X, area=area, intercept=True)
