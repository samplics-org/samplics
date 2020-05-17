import numpy as np
import pandas as pd

import samplics 
from samplics.sae import EblupAreaModel


milk_exp = pd.read_csv("../../../datasets/docs/expenditure_on_milk.csv")

nb_obs = 15
print(f"\nFirst {nb_obs} observations of the Milk Expendure dataset\n")
milk_exp.tail(nb_obs)


area = milk_exp["small_area"]
yhat = milk_exp["direct_est"]
X = pd.get_dummies(milk_exp["major_area"],drop_first=True)
sigma_e = milk_exp["std_error"]

## REML method
fh_model_reml = EblupAreaModel(method="REML")
fh_model_reml.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, tol=1e-8,
)

print(f"\nThe estimated fixed effects are: {fh_model_reml.fixed_effects}")
print(f"\nThe estimated standard error of the area random effects is: {fh_model_reml.re_std}")
print(f"\nThe convergence statistics are: {fh_model_reml.convergence}")
print(f"\nThe goodness of fit statistics are: {fh_model_reml.goodness}\n")


fh_model_reml.predict(
    X=X, area=area,
)

import pprint
pprint.pprint(fh_model_reml.area_est)


milk_est = fh_model_reml.to_dataframe()
print(milk_est)



def dict_to_dataframe(col_names, *args):
    
    values = []
    for k, arg in enumerate(args):
        if not isinstance(arg, dict):
            raise AssertionError("All input parameters must be dictionaries with the same keys.")

        values.append(list(arg.values()))
        
    values_df = pd.DataFrame(values,).T
    values_df.insert(0, "0", list(args[0].keys()))
    values_df.columns = col_names
    
    return values_df


area_est = fh_model_reml.area_est
area_mse = fh_model_reml.area_mse

est_data = dict_to_dataframe(["area", "estimate", "mse"], area_est, area_mse)

print(est_data)






