#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def table_loader(filepath, counties):
    with open(filepath, "r") as f:
        for no, line in enumerate(f.readlines()):
            if no == 0:
                meta       = line.strip("\n").split(",")
                nweeks     = len(meta) - 2
                counts_mat = np.zeros((len(counties), nweeks))
                dates      = [ date.strip("").split("_")[1] for date in meta[1:-1] ] 
                continue
            data   = line.strip("\n").split(",")
            fips   = data[-1].strip("")
            counts = np.array(data[1:nweeks+1])
            counts_mat[counties.index(fips), :] = counts
        return counts_mat, dates

def merge_counties_by_states(data, states, counties, c2s):
    merged_data = []
    for s, cpos in zip(states, c2s):
        county_ind = np.where(cpos)[0]
        merged_data.append(data[county_ind, :].sum(0))
    return np.stack(merged_data, 0)