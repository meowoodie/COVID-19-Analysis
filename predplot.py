#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import table_loader, merge_counties_by_states, statewise_baselines_loader

def case_state_linechart(statename, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-est-case/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='green', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[1], linewidth=4, color="green", alpha=1, linestyle=":", label="STVA")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=12) # loc='upper left'
        plt.title("Confirmed cases in %s" % statename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def death_state_est_linechart(statename, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-est-death/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[1], linewidth=4, color="#AE262A", alpha=2, linestyle=":", label="STVA")
        ax.plot(x, data[2], linewidth=2, color="blue", alpha=.5, linestyle=":", label=r"STVA$-$mobility")
        ax.plot(x, data[3], linewidth=2, color="green", alpha=.5, linestyle=":", label=r"STVA$-$census")
        ax.plot(x, data[4], linewidth=2, color="gray", alpha=.5, linestyle=":", label=r"STVA$-$spatial")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=12, loc='upper left')
        plt.title("Deaths in %s" % statename)
        fig.tight_layout() # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def death_state_pred_linechart(statename, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-pred-death/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))

        # ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#4C9900', alpha=0.2, interpolate=True, label="Real")
        # ax.plot(x, data[1], linewidth=3, color="#4C9900", alpha=1, label="STVA")
        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[1], linewidth=3, color="#AE262A", alpha=1, label="STVA")
        ax.plot(x, data[2], linewidth=2, color="blue", alpha=.5, linestyle="--", label="UT")
        ax.plot(x, data[3], linewidth=2, color="green", alpha=.5, linestyle="--", label="LANL")
        ax.plot(x, data[4], linewidth=2, color="purple", alpha=.5, linestyle="--", label="MOBS")
        ax.plot(x, data[5], linewidth=2, color="orange", alpha=.5, linestyle="--", label="IHME")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=12, loc='upper left')
        plt.title("Deaths in %s" % statename)
        fig.tight_layout() # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def death_county_pred_linechart(countyname, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-pred-death/%s.pdf" % countyname) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))

        # ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#4C9900', alpha=0.2, interpolate=True, label="Real")
        # ax.plot(x, data[1], linewidth=3, color="#4C9900", alpha=1, label="STVA")
        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[1], linewidth=3, color="#AE262A", alpha=1, label="STVA")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=12, loc='upper left')
        plt.title("Deaths in %s" % countyname)
        fig.tight_layout() # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)



if __name__ == "__main__":

    # META DATA
    hubnames   = [
        "Manhattan Borough (New York), NY", "Suffolk County (Boston), MA", "Cook County (Chicago), IL", 
        "King County (Seattle), WA", "Miami-dade County (Miami), FL", "Harris County (Houston), TX", 
        "Fulton County (Atlanta), GA", "Cobb County, GA", "Dekalb County, GA"]
    hubs       = [
        "36061", "36103", "17031", 
        "53033", "12086", "48201", 
        "13121", "13067", "13089"]
    counties   = np.load("mat/counties.npy").tolist()
    states     = np.load("mat/states.npy").tolist()
    c2s        = np.load("mat/state_to_county.npy")
    real_dates = np.load("mat/weeks_1-17.npy")                # [ n_weeks ]
  
    # DATA 
    real_death      = np.load("mat/death_1-17.npy")           # [ n_weeks, n_counties ]
    pred_death      = np.load("mat/onestep/onestep_relu.npy") # [ 15, n_counties ]
    est_death       = np.load("mat/insample/model.npy")  
    real_case       = np.load("mat/ConfirmedCases_1-17.npy")  # [ n_weeks, n_counties ]
    est_case        = np.load("mat/insample/cases.npy")
    # est_death       = np.load("predictions/in-sample-deaths.npy")  
    est_death_nocov = np.load("mat/insample/no_cov.npy")
    est_death_nomob = np.load("mat/insample/no_mob.npy")
    est_death_ar    = np.load("mat/insample/ar.npy").transpose()
    print(est_death_ar.shape)

    # make negative values zero
    est_death[est_death<0]             = 0
    est_death_nocov[est_death_nocov<0] = 0
    est_death_nomob[est_death_nomob<0] = 0

    state_real_death      = merge_counties_by_states(real_death, states, counties, c2s)     # [ n_weeks, n_states ]
    state_pred_death      = merge_counties_by_states(pred_death, states, counties, c2s)      
    state_est_death       = merge_counties_by_states(est_death, states, counties, c2s)
    state_est_death_nocov = merge_counties_by_states(est_death_nocov, states, counties, c2s)
    state_est_death_nomob = merge_counties_by_states(est_death_nomob, states, counties, c2s)
    state_est_death_ar    = merge_counties_by_states(est_death_ar, states, counties, c2s)
    preds                 = statewise_baselines_loader()

    state_real_case       = merge_counties_by_states(real_case, states, counties, c2s)      # [ n_weeks, n_states ]
    state_est_case        = merge_counties_by_states(est_case, states, counties, c2s)

    print(state_real_death.shape, state_est_death.shape, state_est_death_nocov.shape, state_est_death_nomob.shape)

    # # IN-SAMPLE for cases
    # # nationwide
    # nweeks = 45
    # dates  = real_dates[-nweeks:]
    # data   = [ 
    #     real_case.sum(1)[-nweeks:],
    #     est_case.sum(1)[-nweeks-1:-1]
    # ]
    # case_state_linechart("the U.S.", data, dates)
    # # statewide
    # for i, state in enumerate(states):
    #     data = [ 
    #         state_real_case[-nweeks:, i],
    #         state_est_case[-nweeks-1:-1, i]
    #     ]
    #     case_state_linechart(state, data, dates)

    # IN-SAMPLE
    # nationwide
    nweeks = 45
    dates  = real_dates[-nweeks:]
    data   = [ 
        real_death.sum(1)[-nweeks:],
        state_est_death.sum(1)[-nweeks-1:-1],
        state_est_death_nomob.sum(1)[-nweeks-1:-1],
        state_est_death_nocov.sum(1)[-nweeks-1:-1],
        state_est_death_ar.sum(1)[-nweeks-1:-1]
    ]
    death_state_est_linechart("the U.S.", data, dates)
    # statewide
    for i, state in enumerate(states):
        data = [ 
            state_real_death[-nweeks:, i],
            state_est_death[-nweeks-1:-1, i],
            state_est_death_nomob[-nweeks-1:-1, i],
            state_est_death_nocov[-nweeks-1:-1, i],
            state_est_death_ar[-nweeks-1:-1, i]
        ]
        death_state_est_linechart(state, data, dates)

    # # OUT-OF-SAMPLE
    # # nationwide
    # nweeks = 15
    # dates  = real_dates[-nweeks:]
    # data   = [ 
    #     real_death.sum(1)[-nweeks:],
    #     pred_death.sum(1)[-nweeks:],
    #     preds["UT"].sum(1)[-nweeks-1:-1],
    #     preds["LANL"].sum(1)[-nweeks-1:-1],
    #     preds["MOBS"].sum(1)[-nweeks-1:-1],
    #     preds["IHME"].sum(1)[-nweeks-1:-1],
    # ]
    # death_state_pred_linechart("the U.S.", data, dates)
    # # statewide
    # for i, state in enumerate(states):
    #     data = [ 
    #         state_real_death[-nweeks:, i],
    #         state_pred_death[-nweeks:, i],
    #         preds["UT"][-nweeks-1:-1, i],
    #         preds["LANL"][-nweeks-1:-1, i],
    #         preds["MOBS"][-nweeks-1:-1, i],
    #         preds["IHME"][-nweeks-1:-1, i],
    #     ]
    #     death_state_pred_linechart(state, data, dates)
    # # countywise
    # for fips, countyname in zip(hubs, hubnames):
    #     ind  = counties.index(fips)
    #     data = [
    #         real_death[-nweeks:, ind],
    #         pred_death[-nweeks:, ind]
    #     ]
    #     death_county_pred_linechart(countyname, data, dates)

