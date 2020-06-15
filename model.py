#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from scipy.sparse import linalg, csr_matrix

#--------------------------------------------------------------------------
#
# DATA MATRICES
#
#--------------------------------------------------------------------------

# - confirmed cases in week t county i 
#   shape: [ n_weeks, n_counties ]
X  = np.load("/Users/woodie/Desktop/processed_data/conf_cases.npy")
Y  = np.load("/Users/woodie/Desktop/processed_data/icu.npy")
Yp = np.load("/Users/woodie/Desktop/processed_data/death.npy")

#--------------------------------------------------------------------------
#
# META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# - Number of lags
p   = 2
# - list of all counties
#   shape: [ n_counties ]
I   = np.load("/Users/woodie/Desktop/processed_data/counties.npy").tolist()
# - list of all weeks 
#   shape: [ n_weeks - p ]
T   = np.arange(p, X.shape[0]).tolist()
# - adjacency matrix
#   shape: [ n_counties, n_counties ]
adj = np.load("/Users/woodie/Desktop/processed_data/adjacency_matrix.npy")
# - other configurations
n_weeks    = len(T)
n_counties = len(I)

#--------------------------------------------------------------------------
#
# CONSTRUCT COVID MODEL
#
#--------------------------------------------------------------------------

print("[%s] start preparing data..." % arrow.now())
# - response variables
#   shape: [ n_weeks x n_counties ]
y = X[p:, :].reshape(-1)
# - explanatory variables
#   shape: [ n_weeks x n_counties, p x n_counties x n_counties ]
rows, cols, data = [], [], []
for t in T:
    for i in I:
        ti    = T.index(t)           # index of week t   
        ii    = I.index(i)           # index of county i 
        jis   = adj[ii].nonzero()[0] # index of neighboring counties j
        taus  = np.arange(t-p, t)    # past week t-p < tau < t
        # constructing one row of x
        rows += [ ti * n_counties + ii 
            for _, _ in enumerate(taus) for _ in jis ]
        cols += [ taui * n_counties * n_counties + ii * n_counties + ji    
            for taui, _ in enumerate(taus) for ji in jis ]
        data += [ X[tau, ji]            
            for tau in taus for ji in jis ]

x = csr_matrix(
    (data, (rows, cols)), 
    shape=(n_weeks * n_counties, p * n_counties * n_counties))

print("[%s] start fitting model..." % arrow.now())
# data standardization
scaler = StandardScaler(with_mean=False)
scaler.fit(x)
x      = scaler.transform(x)

# ridge regression
regr = Ridge(alpha=.0)
regr.fit(x, y)
beta = regr.coef_

# save results
beta = beta.reshape(p, n_counties, n_counties)
np.save("ridge-beta.npy", beta)