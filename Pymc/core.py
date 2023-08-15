# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:56:58 2023

@author: jczeng
"""

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm




data=pd.read_feather()
Y=data['']
with pm.Model() as model:
    like = pm.Normal("Y_obs", mu=Y.mean(), sigma=np.std(Y), observed=Y)
    idata = pm.sample(cores=7,chains=7)
az.plot_trace(idata, combined=True);
az.plot_energy(idata)
sm1=az.summary(idata, round_to=2)
ppc_trace = pm.sample_posterior_predictive(idata, model=model)
sm2=az.summary(ppc_trace, round_to=2)


