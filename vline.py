import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import table_loader, merge_counties_by_states

def case_state_linechart(statename, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-confirm/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='green', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[1], linewidth=3, color="green", alpha=1, label="In-sample estimation")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=20) # loc='upper left'
        plt.title("Confirmed cases in %s" % statename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def death_state_linechart(statename, data, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    x      = np.arange(len(data[0]))
    ground = np.zeros(len(data[0]))

    with PdfPages("result-death/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.fill_between(x, data[0], ground, where=data[0] >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Real")
        ax.plot(x, data[2], linewidth=3, color="#AE262A", alpha=1, label="In-sample estimation")
        ax.plot(x, data[1], linewidth=3, color="#1A5A98", alpha=1, linestyle='--', label="One-week ahead prediction")

        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers per week")
        plt.legend(fontsize=20) # loc='upper left'
        plt.title("Deaths in %s" % statename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)
   


if __name__ == "__main__":

    # META DATA
    counties     = np.load("predictions/processed_data/counties.npy").tolist()
    states       = np.load("predictions/processed_data/states.npy").tolist()
    c2s          = np.load("predictions/processed_data/state_to_county.npy")

    # DEATH MODEL

    # DATA
    real_death, real_dates = table_loader("predictions/deaths.csv", counties)
    pred_death, pred_dates = table_loader("predictions/predicted-deaths.csv", counties)
    recv_death, recv_dates = table_loader("predictions/recovered-deaths.csv", counties)

    real_death = real_death[:, 11:]
    recv_death = recv_death[:, 11:]
    real_dates = real_dates[11:]

    print(real_death.shape)
    print(recv_death.shape)
    print(pred_death.shape)
    print(len(real_dates))

    state_real_death = merge_counties_by_states(real_death, states, counties, c2s)
    state_pred_death = merge_counties_by_states(pred_death, states, counties, c2s)
    state_recv_death = merge_counties_by_states(recv_death, states, counties, c2s)

    # nationwide
    nweeks = len(pred_dates)
    data   = [ 
        real_death.sum(0)[:nweeks], 
        pred_death.sum(0)[:nweeks],
        recv_death.sum(0)[:nweeks]
    ]
    death_state_linechart("the U.S.", data, pred_dates)

    # statewide
    for i, state in enumerate(states):
        nweeks = len(pred_dates)
        data   = [ 
            state_real_death[i][:nweeks], 
            state_pred_death[i][:nweeks],
            state_recv_death[i][:nweeks]
        ]
        death_state_linechart(state, data, pred_dates)



    # CASE MODEL

    # DATA
    real_case, real_dates = table_loader("predictions/cases.csv", counties)
    recv_case, recv_dates = table_loader("predictions/recovered-cases.csv", counties)

    print(real_case.shape)
    print(recv_case.shape)
    print(len(real_dates))

    state_real_case = merge_counties_by_states(real_case, states, counties, c2s)
    state_recv_case = merge_counties_by_states(recv_case, states, counties, c2s)

    # nationwide
    nweeks = len(recv_dates)
    data   = [ 
        real_case.sum(0)[:nweeks], 
        recv_case.sum(0)[:nweeks],
    ]
    case_state_linechart("the U.S.", data, recv_dates)

    # statewide
    for i, state in enumerate(states):
        nweeks = len(recv_dates)
        data   = [ 
            state_real_case[i][:nweeks], 
            state_recv_case[i][:nweeks]
        ]
        case_state_linechart(state, data, recv_dates)


