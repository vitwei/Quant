# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:54:13 2023

@author: jczeng
"""
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import pyplot as plt

ar1_data=pd.read_feather()

priors = {
    "coefs": {"mu": [10, 0.2], "sigma": [0.1, 0.1], "size": 2},
    "sigma": 8,
    "init": {"mu": 9, "sigma": 0.1, "size": 1},
}

t_data = list(range(len(ar1_data)))
## Initialise the model
with pm.Model() as AR:
## Add the time interval as a mutable coordinate to the model to allow for future predictions
    AR.add_coord("obs_id", t_data, mutable=True)
    ## Data containers to enable prediction
    t = pm.MutableData("t", t_data, dims="obs_id")
    y = pm.MutableData("y", ar1_data, dims="obs_id")
    # The first coefficient will be the constant term but we need to set priors for each coefficient in the AR process
    coefs = pm.Normal("coefs", 0,size=2)
    sigma = pm.HalfNormal("sigma", 0.01)
    # We need one init variable for each lag, hence size is variable too
    init = pm.Normal.dist(
        mu=0, sigma=0.01, size=1
    )
    # Steps of the AR model minus the lags required
    likelihood = pm.AR(
        "ar",
        coefs,
        sigma=sigma,
        init_dist=init,
        constant=True,#constant will be occupy one size
        steps=t.shape[0]-1,
        dims="obs_id",
    )

    # The Likelihood
    outcome = pm.Normal("likelihood", mu=likelihood, sigma=sigma, observed=y, dims="obs_id")
    ## Sampling
    #idata_ar = pm.sample_prior_predictive()
    idata_ar=pm.sample(2000,tune=4000,random_seed=100, target_accept=0.95)
    #idata_ar.extend(pm.sample(2000,tune=4000,random_seed=100, target_accept=0.95))
    idata_ar.extend(pm.sample_posterior_predictive(idata_ar))
    az.plot_energy(idata_ar)
    #predict
with AR:
    # change the value and shape of the data
    pm.set_data(
        {
            "t": [],
            # use dummy values with the same shape:
            "y": [0,0,0],
        },
        coords={"obs_id": [1454, 1455, 1456,1457]},
    )  
    pre=(pm.sample_posterior_predictive(idata_ar))
    sm2=az.summary(pre)
    pvalue=pre.posterior_predictive['likelihood'].mean(dim=["draw", "chain"]).values
    
