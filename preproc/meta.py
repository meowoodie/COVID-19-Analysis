#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

uscountygeo = None

with open("data/us_counties_20m_topo.json", "r") as fgeo:
    uscountygeo = json.loads(fgeo.read())