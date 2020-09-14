#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import torch
import folium
import branca
import numpy as np
# import pandas as pd
# import selenium.webdriver
from preproc import uscountygeo
from covid19linear import COVID19linear
from scipy.sparse import csr_matrix

#--------------------------------------------------------------------------
#
# META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# - Number of lags
p = 2
# - list of all counties
#   shape: [ n_counties ]
hubID     = ["36005", "6075", "13121", "53033", "12086", "17031", "6037", "48201", "48113", "11001"]
hubNames  = ["NYC", "SF", "ATL", "SEA", "MIA", "CHI", "LA", "HST", "DAL", "DC"]
I         = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/predictions/processed_data/counties.npy").tolist()

# Distance matrix for counties
distance = np.sqrt(np.load("mat/distance.npy")) # [ 3144, 3144 ]
adj      = np.load("mat/adjacency_matrix.npy")  # [ 3144, 3144 ]

n_counties   = 3144
n_mobility   = 6
n_covariates = 2

# Coefficients
model = COVID19linear(
    p=p, adj=adj, dist=distance,
    n_counties=n_counties, n_mobility=n_mobility, n_covariates=n_covariates)
# model.load_state_dict(torch.load("fitted_model/new-model-5e-1-nolog-1e38norm-clipper.pt"))
model.load_state_dict(torch.load("fitted_model/new-model-5e2.pt"))
val   = model.A_nonzero[0].detach().numpy()
x, y  = np.where(adj == 1)
X     = csr_matrix((val, (x, y)), shape=(n_counties, n_counties)).toarray()

print(model.mu.detach().numpy())
print(model.nu.detach().numpy())
print(model.upsilon.detach().numpy())
print(model.zeta.detach().numpy())

# import matplotlib.pyplot as plt
# xshow = X.flatten()[np.where(X.flatten() > 0)[0]]
# plt.hist(np.exp(xshow / 10000), bins=20)
# plt.show()

# xshow = X.flatten()[np.where(X.flatten() < 0)[0]]
# plt.hist(np.exp(- xshow / 10000), bins=20)
# plt.show()

# for _fips, _name in zip(hubID, hubNames):
#     print(_name)
#     _id   = I.index(_fips)
#     # X     = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/coef/H.npy", allow_pickle=True)
#     # X0    = X[0] # lag 1
#     coef0 = X[:, _id] # coef at lag 1
#     filename = "A-coef0-%s" % _name

#     #--------------------------------------------------------------------------
#     #
#     # PLOT ON MAP
#     #
#     #--------------------------------------------------------------------------

#     val4county = coef0

#     # take exp
#     negcoefids = np.where(val4county <= 0)[0]
#     poscoefids = np.where(val4county > 0)[0]
#     val4county[poscoefids] = np.exp(val4county[poscoefids] / 10000) 
#     val4county[negcoefids] = - np.exp(- val4county[negcoefids] / 10000)

#     print(val4county.min(), val4county.max())
#     # absmax     = max(abs(val4county.min()), abs(val4county.max()))
#     absmax     = 200000
#     colorscale = branca.colormap.LinearColormap(
#         ['red', 'white', 'blue'],
#         vmin=-absmax, vmax=absmax,
#         caption='Coefficients') # Caption for Color scale or Legend

#     def style_function(feature):
#         county = str(int(feature['id'][-5:]))
#         try:
#             data = val4county[I.index(county)]
#         except Exception as e:
#             data = 0
#         return {
#             'fillOpacity': 0.5,
#             'color': 'black',
#             'weight': .1,
#             'fillColor': '#black' if data is None else colorscale(data)
#         }

#     m = folium.Map(
#         location=[38, -95],
#         tiles='cartodbpositron',
#         zoom_start=4,
#         zoom_control=False,
#         scrollWheelZoom=False
#     )

#     folium.TopoJson(
#         uscountygeo,
#         'objects.us_counties_20m',
#         style_function=style_function
#     ).add_to(m)

#     colorscale.add_to(m)

#     m.save("%s.html" % filename)

# # folium documentation
# # https://python-visualization.github.io/folium/quickstart.html
# # export html to png
# # https://github.com/python-visualization/folium/issues/833
# # phantomjs installation
# # https://stackoverflow.com/questions/36993962/installing-phantomjs-on-mac