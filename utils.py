#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import csv

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
    """
    Merge counties' predictions by states.
    """
    merged_data = []
    for s, cpos in zip(states, c2s):
        county_ind = np.where(cpos)[0]
        merged_data.append(data[:, county_ind].sum(1))
    return np.stack(merged_data, 1)

def statewise_baselines_loader(
    states_path="mat/states.npy", 
    baseline_dir="covid-19-baselines", 
    baseline_list=["IHME", "LANL", "MOBS", "UT", "YYG"]):
    """
    Baseline source:
    https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html#ensembleforecast
    https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html#state-forecasts
    """
    def extract_pred(baseline, states, data):
        # extract point estimation for specified baseline from data
        pred = np.zeros(len(states))
        for row in data[1:]:
            trg_baseline = row[0]
            trg_state    = row[4] 
            trg_pred     = float(row[5]) if row[5] != "NA" else 0.
            if baseline in trg_baseline and trg_state in states:
                pred[states.index(trg_state)] = trg_pred
        return pred

    pred_dict = { baseline: [] for baseline in baseline_list }
    states    = np.load(states_path).tolist()
    filelist  = list(os.listdir(baseline_dir))
    for filename in sorted(filelist):
        if filename.endswith(".csv"): 
            # print(os.path.join(baseline_dir, filename))
            with open(os.path.join(baseline_dir, filename), "r") as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                data   = [ row for row in reader ]
                for baseline in baseline_list:
                    pred_dict[baseline].append(extract_pred(baseline, states, data))

    for baseline in baseline_list:
        pred_dict[baseline] = np.stack(pred_dict[baseline], axis=0)
    # print(pred_dict)
    return pred_dict

if __name__ == "__main__":
    statewise_baselines_loader()