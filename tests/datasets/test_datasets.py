from samplics.datasets.datasets import (load_auto, load_birth,
                                        load_county_crop,
                                        load_county_crop_means,
                                        load_expenditure_milk, load_nhanes2,
                                        load_nhanes2brr, load_nhanes2jk,
                                        load_nmihs, load_psu_frame,
                                        load_psu_sample, load_ssu_sample)


def test_loading_psu_frame():
    psu_frame = load_psu_frame()

    assert psu_frame["name"] == "PSU Frame"
    assert psu_frame["description"] == "A simulated census data."
    assert psu_frame["nrows"] == 100
    assert psu_frame["ncols"] == 5
    assert list(psu_frame["data"].columns) == [
        "cluster",
        "region",
        "number_households_census",
        "cluster_status",
        "comment",
    ]


# breakpoint()
def test_loading_psu_sample():
    psu_sample = load_psu_sample()

    assert psu_sample["name"] == "PSU Sample"
    assert psu_sample["description"] == "The PSU sample obtained from the simulated PSU frame."
    assert psu_sample["nrows"] == 10
    assert psu_sample["ncols"] == 3
    assert list(psu_sample["data"].columns) == ["cluster", "region", "psu_prob"]


def test_loading_ssu_sample():
    ssu_sample = load_ssu_sample()

    assert ssu_sample["name"] == "SSU Sample"
    assert ssu_sample["description"] == "The SSU sample obtained from the simulated SSU frame."
    assert ssu_sample["nrows"] == 150
    assert ssu_sample["ncols"] == 3
    assert list(ssu_sample["data"].columns) == ["cluster", "household", "ssu_prob"]


def test_loading_nhanes2_subset():
    nhanes2 = load_nhanes2()

    assert nhanes2["name"] == "NHANES II Subsample"
    assert (
        nhanes2["description"]
        == "A subset of NHANES II data. This file is not meant to be representative of NHANES II. It is just an subset to illustrate the syntax in this tutorial."
    )
    assert nhanes2["nrows"] == 10337
    assert nhanes2["ncols"] == 8
    assert list(nhanes2["data"].columns) == [
        "stratid",
        "psuid",
        "race",
        "highbp",
        "highlead",
        "zinc",
        "diabetes",
        "finalwgt",
    ]


def test_loading_nnhanes2brr_subset():
    nhanes2brr_subset = load_nhanes2brr()

    assert nhanes2brr_subset["name"] == "NHANES II Subsample with bootstrap weights"
    assert (
        nhanes2brr_subset["description"]
        == "A subset of NHANES II data with bootstrap weights. This file is not meant to be representative of NHANES II. It is just an subset to illustrate the syntax in this tutorial."
    )
    assert nhanes2brr_subset["nrows"] == 1347
    assert nhanes2brr_subset["ncols"] == 35


def test_loading_nnhanes2jk_subset():
    nhanes2jk_subset = load_nhanes2jk()

    assert nhanes2jk_subset["name"] == "NHANES II Subsample with jackknife weights"
    assert (
        nhanes2jk_subset["description"]
        == "A subset of NHANES II data with jackknife weights. This file is not meant to be representative of NHANES II. It is just an subset to illustrate the syntax in this tutorial."
    )
    assert nhanes2jk_subset["nrows"] == 887
    assert nhanes2jk_subset["ncols"] == 65


def test_loading_nmihs_subset():
    nmihs_subset = load_nmihs()

    assert nmihs_subset["name"] == "NMIHS Subsample"
    assert (
        nmihs_subset["description"]
        == "A subset of nmihs data. This file is not meant to be representative of nmihs. It is just an subset to illustrate the syntax in this tutorial."
    )
    assert nmihs_subset["nrows"] == 603
    assert nmihs_subset["ncols"] == 52


def test_loading_auto():
    auto = load_auto()

    assert auto["name"] == "Auto Sample"
    assert auto["description"] == "The Auto sample data."
    assert auto["nrows"] == 74
    assert auto["ncols"] == 4
    assert list(auto["data"].columns) == ["mpg", "foreign", "y1", "y2"]


def test_loading_birth():
    birth = load_birth()

    assert birth["name"] == "Birth Sample"
    assert birth["description"] == "The Birth sample data."
    assert birth["nrows"] == 956
    assert birth["ncols"] == 4
    assert list(birth["data"].columns) == ["region", "agecat", "birthcat", "pop"]


def test_loading_county_crop_areas():
    county_crop = load_county_crop()

    assert county_crop["name"] == "County Crop Sample"
    assert county_crop["description"] == "The County Crop Areas sample data."
    assert county_crop["nrows"] == 37
    assert county_crop["ncols"] == 5
    assert list(county_crop["data"].columns) == [
        "county_id",
        "corn_area",
        "soybeans_area",
        "corn_pixel",
        "soybeans_pixel",
    ]


def test_loading_county_crop_areas_means():
    county_crop_means = load_county_crop_means()

    assert county_crop_means["name"] == "County Crop Area Means"
    assert county_crop_means["description"] == "The County Crop Area Means data."
    assert county_crop_means["nrows"] == 12
    assert county_crop_means["ncols"] == 5
    assert list(county_crop_means["data"].columns) == [
        "county_id",
        "samp_segments",
        "pop_segments",
        "ave_corn_pixel",
        "ave_soybeans_pixel",
    ]


def test_loading_expenditure_on_milk():
    expenditure_milk = load_expenditure_milk()

    assert expenditure_milk["name"] == "Expenditure on Milk"
    assert expenditure_milk["description"] == "The expenditure on milk data."
    assert expenditure_milk["nrows"] == 43
    assert expenditure_milk["ncols"] == 6
    assert list(expenditure_milk["data"].columns) == [
        "major_area",
        "small_area",
        "samp_size",
        "direct_est",
        "std_error",
        "coef_var",
    ]
