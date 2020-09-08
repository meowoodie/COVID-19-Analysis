#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import folium
import branca
import numpy as np
# import pandas as pd
# import selenium.webdriver
from preproc import uscountygeo

#--------------------------------------------------------------------------
#
# META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# - Number of lags
p = 18
# - list of all counties
#   shape: [ n_counties ]
hubID     = ["36005", "6075", "13121", "53033", "12086", "17031", "6037", "48201", "48113", "11001"]
hubNames  = ["NYC", "SF", "ATL", "SEA", "MIA", "CHI", "LA", "HST", "DAL", "DC"]
I         = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/predictions/processed_data/counties.npy").tolist()

for _fips, _name in zip(hubID, hubNames):
    print(_name)
    _id   = I.index(_fips)
    X     = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/coef/H.npy", allow_pickle=True)
    X0    = X[0] # lag 1
    # X1    = X[1] # lag 2
    coef0 = X0[:, _id].toarray().flatten() # coef at lag 1
    # coef1 = X0[:, _id].toarray().flatten() # coef at lag 2
    filename = "H-coef-%s-lag0" % _name

    #--------------------------------------------------------------------------
    #
    # PLOT ON MAP
    #
    #--------------------------------------------------------------------------

    val4county = coef0

    # take log
    # negcoefids = np.where(val4county <= 0)[0]
    # poscoefids = np.where(val4county > 0)[0]
    # val4county[poscoefids] = np.log(val4county[poscoefids])
    # val4county[negcoefids] = - np.log(- val4county[negcoefids] + 1e-10)

    print(val4county.min(), val4county.max())
    absmax     = max(abs(val4county.min()), abs(val4county.max()))
    colorscale = branca.colormap.LinearColormap(
        ['red', 'white', 'blue'],
        vmin=-absmax, vmax=absmax,
        caption='Coefficients') # Caption for Color scale or Legend

    def style_function(feature):
        county = str(int(feature['id'][-5:]))
        try:
            data = val4county[I.index(county)]
        except Exception as e:
            data = 0
        return {
            'fillOpacity': 0.5,
            'weight': 0,
            'fillColor': '#black' if data is None else colorscale(data)
        }

    m = folium.Map(
        location=[38, -95],
        tiles='cartodbpositron',
        zoom_start=5
    )

    folium.TopoJson(
        uscountygeo,
        'objects.us_counties_20m',
        style_function=style_function
    ).add_to(m)

    colorscale.add_to(m)

    m.save("%s.html" % filename)

# folium documentation
# https://python-visualization.github.io/folium/quickstart.html
# export html to png
# https://github.com/python-visualization/folium/issues/833
# phantomjs installation
# https://stackoverflow.com/questions/36993962/installing-phantomjs-on-mac