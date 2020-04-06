#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from collections import defaultdict

# data initialization
covid19bycounty = {}       # get cumsum value: covid19bycounty[fips].cumsum()
covid19bydate   = {}       

# initialize the county-wise dataframe of covid-19  
sdate = date(2020, 1, 1)   # start date
edate = date(2020, 4, 4)   # end date
delta = edate - sdate      # as timedelta
days  = [ str(sdate + timedelta(days=i)) for i in range(delta.days + 1) ]
cols  = ["confirmed", "death"]
init  = np.zeros((len(days), len(cols)))
df0   = pd.DataFrame(init, index=days, columns=cols)

# FIPS dictionary initializer `{LONGNAME: FIPS}`
with open("data/us-fips.csv", "r") as fdict:
    data = [ line.strip("\n").split(",") for line in list(fdict)[1:] ]
    fips = { d[4].lower(): d[0] for d in data }
# covid-19 by county time series data from 1-point-3-acres
with open("data/us-1point3acres-april-5.csv", "r") as f1p3a:
    missedq = []
    for line in list(f1p3a)[1:]:
        d = line.strip("\n").split(",")
        date, state, county, confirmed, death = d[1], d[2], d[3], d[4], d[5]
        query = "%s county %s" % (county.lower(), state.lower())
        try:
            _id = fips[query]
            if _id not in covid19bycounty.keys():
                covid19bycounty[_id] = df0.copy()
            covid19bycounty[_id].at[date, "confirmed"] += int(confirmed)
            covid19bycounty[_id].at[date, "death"]     += int(death)
        except Exception as e:
            missedq.append(query)

# # covid-19 by county time series data from nytimes
# # https://github.com/COVIDExposureIndices/COVIDExposureIndices
# url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
# nytimesdf = requests.get(url).text.split("\n")
# for line in nytimesdf[1:]:
#     d = line.strip("").split(",")
#     date, county, state, fips, confirmed, death = d
#     if fips not in covid19bycounty.keys():
#         covid19bycounty[fips] = df0.copy()
#     covid19bycounty[fips].at[date, "confirmed"] += int(confirmed)
#     covid19bycounty[fips].at[date, "death"]     += int(death)

# initialize the day-wise dataframe of covid-19 
counties = list(covid19bycounty.keys())
counties.sort()
cols     = ["confirmed", "death"]
init     = np.zeros((len(counties), len(cols)))
df1      = pd.DataFrame(init, index=counties, columns=cols)
# covid-19 by date time series data from 1-point-3-acres
covid19cumsum = { county: covid19bycounty[county].cumsum() for county in counties }
covid19bydate = { date: df1.copy() for date in days }
for date in days:
    for county in counties:
        covid19bydate[date].at[county, "confirmed"] = covid19cumsum[county].at[date, "confirmed"]
        covid19bydate[date].at[county, "death"]     = covid19cumsum[county].at[date, "death"]

# maximum confirmed and death
maxconf  = covid19bydate[days[-1]]['confirmed'].max()
maxdeath = covid19bydate[days[-1]]['death'].max()



if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    print(list(set(missedq)))
    print(len(set(missedq)))
    print(covid19bycounty["13121"].cumsum())
    # print(covid19bydate[days[-1]])