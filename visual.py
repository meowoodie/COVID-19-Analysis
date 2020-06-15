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
I         = np.load("/Users/woodie/Desktop/processed_data/counties.npy").tolist()

# beta      = np.load("ridge-beta.npy")
# newyorkid = I.index("53033")
# X         = beta[:, newyorkid, :]
# X         = X - X.min()
# filename  = "covid19-nyc-beta-week"

X        = np.load("/Users/woodie/Desktop/processed_data/conf_cases.npy")
X[1:, :] = X[1:, :] - X[:-1, :]
print((X < 0).sum())
X[X < 0] = 0
filename = "covid19-confirmed-week"

#--------------------------------------------------------------------------
#
# PLOT ON MAP
#
#--------------------------------------------------------------------------

for tau in range(p):
    print(tau)

    val4county = X[tau]
    # print(beta.min(), beta.max())
    colorscale = branca.colormap.linear.YlOrRd_09.scale(
        np.log(1e-1), np.log(val4county.max()))

    def style_function(feature):
        county = str(int(feature['id'][-5:]))
        try:
            # data = covid19.at[county, "confirmed"]
            data = np.log(val4county[I.index(county)])
            # data = np.log(1e-1) if data <= 0. else np.log(data)
        except Exception as e:
            data = np.log(1e-1)
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

    m.save("%s-%s.html" % (filename, tau))

# driver = selenium.webdriver.PhantomJS()
# driver.set_window_size(4000, 3000)  # choose a resolution
# driver.get("result/covid19-confirmed-%s.html" % date)
# # You may need to add time.sleep(seconds) here
# time.sleep(1000)
# driver.save_screenshot("result/covid19-confirmed-%s.png" % date)

# print(np.log(1e-1), np.log(maxconf))

# folium documentation
# https://python-visualization.github.io/folium/quickstart.html
# export html to png
# https://github.com/python-visualization/folium/issues/833
# phantomjs installation
# https://stackoverflow.com/questions/36993962/installing-phantomjs-on-mac