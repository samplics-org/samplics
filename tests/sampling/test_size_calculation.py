from samplics.sampling import OneSampleSize

## Design effects


deff_calculation = OneSampleSize()


def test_deff_int():
    assert deff_calculation.deff(30, 0.01) == 1.29


def test_deff_float():
    assert deff_calculation.deff(15.5, 0.03) == 1.435


def test_deff_dict():
    m = {"stratum1": 30, "stratum2": 15.5, "stratum3": 50}
    icc = {"stratum1": 0.01, "stratum2": 0.03, "stratum3": 0.10}
    deff = deff_calculation.deff(m, icc)
    assert deff == {"stratum1": 1.29, "stratum2": 1.435, "stratum3": 5.9}


## Wald's method

size_nat_wald = OneSampleSize()


def test_size_nat_wald_basics():
    assert size_nat_wald.parameter == "proportion"
    assert size_nat_wald.method == "wald"
    assert size_nat_wald.stratification == False


def test_size_nat_wald_size():
    size_nat_wald.calculate(0.80, 0.10)
    assert size_nat_wald.samp_size == 62
    assert size_nat_wald.deff_c == 1.0
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.precision == 0.1


def test_size_nat_wald_size_with_deff():
    size_nat_wald.calculate(0.80, 0.10, deff=1.5)
    assert size_nat_wald.samp_size == 93
    assert size_nat_wald.deff_c == 1.5
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.precision == 0.1


def test_size_nat_wald_df():
    size_nat_wald.calculate(0.80, 0.10)
    size_df = size_nat_wald.to_dataframe()
    assert (size_df.columns == ["_target", "_precision", "_samp_size"]).all()


## Wald's method - stratified
size_str_wald = OneSampleSize(parameter="Proportion", method="Wald", stratification=True)

target = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}
precision = {"stratum1": 0.30, "stratum2": 0.10, "stratum3": 0.15}
deff = {"stratum1": 1, "stratum2": 1.5, "stratum3": 2.5}


def test_size_str_wald_basics():
    assert size_str_wald.parameter == "proportion"
    assert size_str_wald.method == "wald"
    assert size_str_wald.stratification == True


def test_size_str_wald_size1():
    size_str_wald.calculate(target, 0.10)
    assert size_str_wald.samp_size["stratum1"] == 19
    assert size_str_wald.samp_size["stratum2"] == 81
    assert size_str_wald.samp_size["stratum3"] == 81


def test_size_str_wald_size2():
    size_str_wald.calculate(0.8, precision)
    assert size_str_wald.samp_size["stratum1"] == 7
    assert size_str_wald.samp_size["stratum2"] == 62
    assert size_str_wald.samp_size["stratum3"] == 28


def test_size_str_wald_size3():
    size_str_wald.calculate(0.8, 0.10, deff)
    assert size_str_wald.samp_size["stratum1"] == 62
    assert size_str_wald.samp_size["stratum2"] == 93
    assert size_str_wald.samp_size["stratum3"] == 154


def test_size_str_wald_size4():
    size_str_wald.calculate(target, precision, deff)
    assert size_str_wald.samp_size["stratum1"] == 3
    assert size_str_wald.samp_size["stratum2"] == 122
    assert size_str_wald.samp_size["stratum3"] == 90


def test_size_str_wald_size5():
    size_str_wald.calculate(0.8, 0.1, 1.5, number_strata=5)
    assert size_str_wald.samp_size["_stratum_1"] == 93
    assert size_str_wald.samp_size["_stratum_2"] == 93
    assert size_str_wald.samp_size["_stratum_3"] == 93
    assert size_str_wald.samp_size["_stratum_4"] == 93
    assert size_str_wald.samp_size["_stratum_5"] == 93


def test_size_str_wald_df():
    size_str_wald.calculate(0.80, 0.10, number_strata=5)
    size_df = size_str_wald.to_dataframe()
    assert size_df.shape[0] == 5
    assert (size_df.columns == ["_stratum", "_target", "_precision", "_samp_size"]).all()


## Fleiss' method

size_nat_fleiss = OneSampleSize(method="fleiss")


def test_size_nat_fleiss_basics():
    assert size_nat_fleiss.parameter == "proportion"
    assert size_nat_fleiss.method == "fleiss"
    assert size_nat_fleiss.stratification == False


def test_size_nat_fleiss_size1a():
    size_nat_fleiss.calculate(0.80, 0.10)
    assert size_nat_fleiss.samp_size == 88
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.precision == 0.1


def test_size_nat_fleiss_size1b():
    size_nat_fleiss.calculate(0.20, 0.10)
    assert size_nat_fleiss.samp_size == 88
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.20
    assert size_nat_fleiss.precision == 0.1


def test_size_nat_fleiss_size2a():
    size_nat_fleiss.calculate(0.95, 0.06)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.95
    assert size_nat_fleiss.precision == 0.06


def test_size_nat_fleiss_size2b():
    size_nat_fleiss.calculate(0.05, 0.06)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.05
    assert size_nat_fleiss.precision == 0.06


def test_size_nat_fleiss_size3():
    size_nat_fleiss.calculate(0.70, 0.03)
    assert size_nat_fleiss.samp_size == 1097
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.70
    assert size_nat_fleiss.precision == 0.03


def test_size_nat_fleiss_size4():
    size_nat_fleiss.calculate(0.85, 0.03)
    assert size_nat_fleiss.samp_size == 663
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.85
    assert size_nat_fleiss.precision == 0.03


def test_size_nat_fleiss_size_with_deff1a():
    size_nat_fleiss.calculate(0.80, 0.10, deff=1.5)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.precision == 0.1


def test_size_nat_fleiss_size_with_deff1b():
    size_nat_fleiss.calculate(0.20, 0.10, deff=1.5)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.20
    assert size_nat_fleiss.precision == 0.1


def test_size_nat_fleiss_size_with_deff2a():
    size_nat_fleiss.calculate(0.95, 0.06, deff=1.5)
    assert size_nat_fleiss.samp_size == 197
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.95
    assert size_nat_fleiss.precision == 0.06


def test_size_nat_fleiss_size_with_deff2b():
    size_nat_fleiss.calculate(0.05, 0.06, deff=1.5)
    assert size_nat_fleiss.samp_size == 197
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.05
    assert size_nat_fleiss.precision == 0.06


def test_size_nat_fleiss_size_with_deff3():
    size_nat_fleiss.calculate(0.70, 0.03, deff=1.5)
    assert size_nat_fleiss.samp_size == 1646
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.70
    assert size_nat_fleiss.precision == 0.03


def test_size_nat_fleiss_size_with_deff4():
    size_nat_fleiss.calculate(0.85, 0.03, deff=1.5)
    assert size_nat_fleiss.samp_size == 994
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.85
    assert size_nat_fleiss.precision == 0.03


def test_size_nat_fleiss_df():
    size_nat_fleiss.calculate(0.80, 0.10)
    size_df = size_nat_fleiss.to_dataframe()
    assert (size_df.columns == ["_target", "_precision", "_samp_size"]).all()


## Fleiss' method - stratified
size_str_fleiss = OneSampleSize(parameter="Proportion", method="Fleiss", stratification=True)

target2 = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}
precision2 = {"stratum1": 0.03, "stratum2": 0.10, "stratum3": 0.05}
deff2 = {"stratum1": 1, "stratum2": 1.5, "stratum3": 2.5}


def test_size_str_fleiss_basics():
    assert size_str_fleiss.parameter == "proportion"
    assert size_str_fleiss.method == "fleiss"
    assert size_str_fleiss.stratification == True


def test_size_str_fleiss_size1():
    size_str_fleiss.calculate(target2, 0.10)
    assert size_str_fleiss.samp_size["stratum1"] == 70
    assert size_str_fleiss.samp_size["stratum2"] == 103
    assert size_str_fleiss.samp_size["stratum3"] == 103


def test_size_str_fleiss_size2():
    size_str_fleiss.calculate(0.8, precision2)
    assert size_str_fleiss.samp_size["stratum1"] == 788
    assert size_str_fleiss.samp_size["stratum2"] == 88
    assert size_str_fleiss.samp_size["stratum3"] == 306


def test_size_str_fleiss_size3():
    size_str_fleiss.calculate(0.8, 0.10, deff2)
    assert size_str_fleiss.samp_size["stratum1"] == 88
    assert size_str_fleiss.samp_size["stratum2"] == 132
    assert size_str_fleiss.samp_size["stratum3"] == 220


def test_size_str_fleiss_size4():
    size_str_fleiss.calculate(target2, precision2, deff2)
    assert size_str_fleiss.samp_size["stratum1"] == 354
    assert size_str_fleiss.samp_size["stratum2"] == 154
    assert size_str_fleiss.samp_size["stratum3"] == 1002


def test_size_str_fleiss_size5():
    size_str_fleiss.calculate(0.8, 0.1, 1.5, number_strata=5)
    assert size_str_fleiss.samp_size["_stratum_1"] == 132
    assert size_str_fleiss.samp_size["_stratum_2"] == 132
    assert size_str_fleiss.samp_size["_stratum_3"] == 132
    assert size_str_fleiss.samp_size["_stratum_4"] == 132
    assert size_str_fleiss.samp_size["_stratum_5"] == 132


def test_size_str_fleiss_df():
    size_str_fleiss.calculate(0.80, 0.10, number_strata=5)
    size_df = size_str_fleiss.to_dataframe()
    assert size_df.shape[0] == 5
    assert (size_df.columns == ["_stratum", "_target", "_precision", "_samp_size"]).all()
