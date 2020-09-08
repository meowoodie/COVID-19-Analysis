#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import folium
import branca
import numpy as np
from preproc import uscountygeo

#--------------------------------------------------------------------------
#
# META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# - list of all counties
#   shape: [ n_counties ]
I    = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/predictions/processed_data/counties.npy").tolist()

X    = np.load("/Users/woodie/Dropbox (GaTech)/workspace/COVID-19-Analysis/coef/V.npy", allow_pickle=True)
coef = X[:, 5]
filename = "V-coef-5"

#--------------------------------------------------------------------------
#
# PLOT ON MAP
#
#--------------------------------------------------------------------------

val4county = coef

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