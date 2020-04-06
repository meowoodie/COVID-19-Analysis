#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import folium
import branca
import numpy as np
import pandas as pd
import selenium.webdriver

from preproc import uscountygeo, covid19bydate, maxdeath, maxconf, days

for date in days:
    print(date)

    colorscale = branca.colormap.linear.YlOrRd_09.scale(np.log(1e-1), np.log(maxconf))
    covid19    = covid19bydate[date]

    def style_function(feature):
        county = str(int(feature['id'][-5:]))
        try:
            data = covid19.at[county, "confirmed"]
            data = np.log(1e-1) if data <= 0. else np.log(data)
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

    m.save("result_confirmed/covid19-confirmed-%s.html" % date)

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