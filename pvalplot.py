#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import branca
import folium
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.sparse import csr_matrix
from matplotlib.backends.backend_pdf import PdfPages
from covid19linear import COVID19linear
from preproc import uscountygeo
from utils import table_loader, merge_counties_by_states, statewise_baselines_loader


def pvalue_spatial_coefficients(model, lag, adj, n_counties, p):
    """
    Calculate p value for spatial coefficients in the model
    """
    # Model evaluation
    model.eval()
    C_hat, D_hat = model(
        C=torch.FloatTensor(C), D=torch.FloatTensor(D), 
        M=torch.FloatTensor(M), cov=torch.FloatTensor(cov))
    C_hat = torch.stack(C_hat, dim=0).detach().numpy()
    D_hat = torch.stack(D_hat, dim=0).detach().numpy()

    x, y  = np.where(adj == 1)
    A     = model.A_nonzero[lag].detach().numpy()
    B     = model.B_nonzero[lag].detach().numpy()
    H     = model.H_nonzero[lag].detach().numpy()
    A     = csr_matrix((A, (x, y)), shape=(n_counties, n_counties)).toarray()
    B     = csr_matrix((B, (x, y)), shape=(n_counties, n_counties)).toarray()
    H     = csr_matrix((H, (x, y)), shape=(n_counties, n_counties)).toarray()

    # Calculate variance of the cases and deaths models
    varC  = ((C_hat - C[p:,:]) ** 2).sum(0)
    varD  = ((D_hat - D[p:,:]) ** 2).sum(0)

    # t statistics
    tA    = A / (np.sqrt(varD / (D.shape[0] - p) * n_counties)) / 1e-8
    tB    = B / (np.sqrt(varC / (C.shape[0] - p) * n_counties)) / 1e-8
    tH    = H / (np.sqrt(varC / (C.shape[0] - p) * n_counties)) / 1e-8

    print(tA.min(), tA.max())

    # p value
    # https://stackoverflow.com/questions/17559897/python-p-value-from-t-statistic
    pvalA = stats.t.sf(np.abs(tA), (D.shape[0] - p) * n_counties) * 2
    pvalB = stats.t.sf(np.abs(tB), (C.shape[0] - p) * n_counties) * 2
    pvalH = stats.t.sf(np.abs(tH), (C.shape[0] - p) * n_counties) * 2

    return pvalA, pvalB, pvalH

def coefficient_us_map(X, filename="pvalue"):
    """
    Visualize coefficients on the map of the US
    """
    # list of all counties
    I         = np.load("mat/counties.npy").tolist()
    # list of all transportation hubs
    hubID     = ["36005", "6075", "13121", "53033", "12086", "17031", "6037", "48201", "48113", "11001"]
    hubNames  = ["NYC", "SF", "ATL", "SEA", "MIA", "CHI", "LA", "HST", "DAL", "DC"]

    for _fips, _name in zip(hubID, hubNames):
        print(_name)
        _id  = I.index(_fips)
        coef = X[:, _id]

        print(coef.min(), coef.max())
        colorscale = branca.colormap.LinearColormap(
            ['green', 'white'],
            vmin=0, vmax=.05,
            caption='p value') # Caption for Color scale or Legend

        def style_function(feature):
            county = str(int(feature['id'][-5:]))
            try:
                data = coef[I.index(county)]
            except Exception as e:
                data = 1.
            return {
                'fillOpacity': 0.5,
                'color': 'black',
                'weight': .1,
                'fillColor': '#black' if data is None else colorscale(data)
            }

        m = folium.Map(
            location=[38, -95],
            tiles='cartodbpositron',
            zoom_start=4,
            zoom_control=False,
            scrollWheelZoom=False
        )

        folium.TopoJson(
            uscountygeo,
            'objects.us_counties_20m',
            style_function=style_function
        ).add_to(m)

        colorscale.add_to(m)

        m.save("result-maps/%s-%s.html" % (filename, _name))


if __name__ == "__main__":

    # confirmed cases and deaths
    C = np.load("mat/ConfirmedCases_1-17.npy") # [ T, counties ]
    D = np.load("mat/death_1-17.npy")          # [ T, counties ]
    # Load covariates
    M      = np.load("mat/mobility_1-17.npy").transpose([2,0,1]) # [ n_mobility, T, counties ]
    pop    = np.load("mat/population.npy")
    over60 = np.load("mat/over60.npy")
    cov    = np.array([pop, over60])                             # [ n_covariates, T, counties ]
    T, n_counties = C.shape
    n_mobility    = M.shape[0]
    n_covariates  = cov.shape[0]
    # Number of lags
    p = 2
    # Distance matrix for counties
    distance = np.sqrt(np.load("mat/distance.npy")) # [ 3144, 3144 ]
    adj      = np.load("mat/adjacency_matrix.npy")  # [ 3144, 3144 ]

    # Model definition
    model = COVID19linear(
        p=p, adj=adj, dist=distance,
        n_counties=n_counties, n_mobility=n_mobility, n_covariates=n_covariates)
    
    model.load_state_dict(torch.load("fitted_model/new-model-5e2.pt"))

    pvalA, pvalB, pvalH = pvalue_spatial_coefficients(model, lag=1, adj=adj, n_counties=n_counties, p=p)
    coefficient_us_map(pvalA, filename="pvalue-A-lag1")
    coefficient_us_map(pvalB, filename="pvalue-B-lag1")
    coefficient_us_map(pvalH, filename="pvalue-H-lag1")