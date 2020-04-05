#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import folium
import branca
import numpy as np
import pandas as pd

from preproc import uscountygeo, covid19bydate, maxdeath, maxconf

colorscale = branca.colormap.linear.YlOrRd_09.scale(np.log(1e-1), np.log(maxconf))
covid19    = covid19bydate["2020-04-04"]

def style_function(feature):
    county = str(int(feature['id'][-5:]))
    try:
        data = covid19.at[county, "death"]
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

m.save("covid19-death.html")

print(np.log(1e-1), np.log(maxconf))