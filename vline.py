import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import table_loader, merge_counties_by_states

def confirm_state_linechart(statename, data, proj, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("result-confirm/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(len(data[0])), data[0], c="#677a04", linewidth=3, linestyle="-", marker='*', markersize=10, label="Real", alpha=.8)
        ax.plot(np.arange(len(data[1])), data[1], c="#cb416b", linewidth=3, linestyle="--", marker='+', markersize=10, label="One-week projection", alpha=1.)
        # ax.plot(np.arange(len(data[2])), data[2], c="#cb416b", linewidth=3, linestyle="--", marker='+', markersize=10, label="Two-weeks projection", alpha=.5)

        ax.plot(np.arange(len(data[0])-1, len(data[0])+len(proj[0])-1), proj[0], c="#cea2fd", linewidth=3, linestyle="--", marker='+', markersize=5, label="One-week projection (future)", alpha=1.)
        # ax.plot(np.arange(len(data[0])-1, len(data[0])+len(proj[1])-1), proj[1], c="#cea2fd", linewidth=3, linestyle="--", marker='+', markersize=5, label="Two-weeks projection (future)", alpha=5.)

        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers")
        plt.legend(loc='upper left',fontsize=13)
        plt.title("Confirmed cases in %s" % statename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)

def death_state_linechart(statename, data, proj, dates):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("result-death/%s.pdf" % statename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(len(data[0])), data[0], c="#677a04", linewidth=3, linestyle="-", marker='*', markersize=10, label="Real", alpha=.8)
        ax.plot(np.arange(len(data[1])), data[1], c="#cb416b", linewidth=3, linestyle="--", marker='+', markersize=10, label="One-week projection", alpha=1.)
        ax.plot(np.arange(len(data[2])), data[2], c="#cb416b", linewidth=3, linestyle="--", marker='+', markersize=10, label="Two-weeks projection", alpha=.5)
        # ax.plot(np.arange(len(data[3])), data[3], c="#cb416b", linewidth=3, linestyle="--", marker='+', markersize=10, label="Three-weeks projection", alpha=.2)

        ax.plot(np.arange(len(data[0])-1, len(data[0])+len(proj[0])-1), proj[0], c="#cea2fd", linewidth=3, linestyle="--", marker='+', markersize=10, label="One-week projection (future)", alpha=1.)
        ax.plot(np.arange(len(data[0])-1, len(data[0])+len(proj[1])-1), proj[1], c="#cea2fd", linewidth=3, linestyle="--", marker='+', markersize=10, label="Two-weeks projection (future)", alpha=.5)
        # ax.plot(np.arange(len(data[0])-1, len(data[0])+len(proj[2])-1), proj[2], c="#cea2fd", linewidth=3, linestyle="--", marker='+', markersize=10, label="Three-weeks projection (future)", alpha=.2)

        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(len(dates)), dates, rotation=90)
        plt.xlabel(r"Dates")
        plt.ylabel(r"Numbers")
        plt.legend(loc='upper left',fontsize=13)
        plt.title("Death toll in %s" % statename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        pdf.savefig(fig)
   


if __name__ == "__main__":

    # META DATA
    counties     = np.load("predictions/processed_data/counties.npy").tolist()
    states       = np.load("predictions/processed_data/states.npy").tolist()
    c2s          = np.load("predictions/processed_data/state_to_county.npy")

    # # DATA
    real_confirm, rcdates = table_loader("predictions/ConfirmedCases.csv", counties)
    real_death, rddates   = table_loader("predictions/Deaths.csv", counties)

    print(rcdates)

    pred_confirm_1week, pc1dates = table_loader("predictions/oneWeekCasesPrediction.csv", counties)
    pred_confirm_2week, pc2dates = table_loader("predictions/twoWeekCasesPrediction.csv", counties)
    pred_death_1week, pd1dates   = table_loader("predictions/oneWeekDeathPrediction.csv", counties)
    pred_death_2week, pd2dates   = table_loader("predictions/twoWeekDeathPrediction.csv", counties)
    pred_death_3week, pd3dates   = table_loader("predictions/threeWeekDeathPrediction.csv", counties)

    sw_real_confirm = merge_counties_by_states(real_confirm, states, counties, c2s)
    sw_pred1w_confirm = merge_counties_by_states(pred_confirm_1week, states, counties, c2s)
    # sw_pred2w_confirm = merge_counties_by_states(pred_confirm_2week, states, counties, c2s)

    # nationwide
    nweeks = len(rcdates)
    data   = [ 
        real_confirm.sum(0)[:nweeks], 
        pred_confirm_1week.sum(0)[:nweeks], 
        # pred_confirm_2week.sum(0)[:nweeks] 
    ]
    proj   = [ 
        pred_confirm_1week.sum(0)[nweeks-1:], 
        # pred_confirm_2week.sum(0)[nweeks-1:] 
    ]
    confirm_state_linechart("the U.S.", data, proj, pc2dates)

    # statewide
    for i, state in enumerate(states):
        nweeks = len(rcdates)
        data   = [ 
            sw_real_confirm[i][:nweeks], 
            sw_pred1w_confirm[i][:nweeks], 
            # sw_pred2w_confirm[i][:nweeks] 
        ]
        proj   = [ 
            sw_pred1w_confirm[i][nweeks-1:], 
            # sw_pred2w_confirm[i][nweeks-1:] 
        ]
        confirm_state_linechart(state, data, proj, pc2dates)

    sw_real_death   = merge_counties_by_states(real_death, states, counties, c2s)
    sw_pred1w_death = merge_counties_by_states(pred_death_1week, states, counties, c2s)
    sw_pred2w_death = merge_counties_by_states(pred_death_2week, states, counties, c2s)
    # sw_pred3w_death = merge_counties_by_states(pred_death_3week, states, counties, c2s)

    # nationwide
    nweeks = len(rddates)
    data   = [ 
        real_death.sum(0)[:nweeks], 
        pred_death_1week.sum(0)[:nweeks], 
        pred_death_2week.sum(0)[:nweeks],
        # pred_death_3week.sum(0)[:nweeks] 
    ]
    proj   = [ 
        pred_death_1week.sum(0)[nweeks-1:], 
        pred_death_2week.sum(0)[nweeks-1:], 
        # pred_death_3week.sum(0)[nweeks-1:] 
    ]
    death_state_linechart("the U.S.", data, proj, pd3dates)

    # statewide
    for i, state in enumerate(states):
        nweeks = len(rcdates)
        data   = [ 
            sw_real_death[i][:nweeks], 
            sw_pred1w_death[i][:nweeks], 
            sw_pred2w_death[i][:nweeks],
            # sw_pred3w_death[i][:nweeks] 
        ]
        proj   = [ 
            sw_pred1w_death[i][nweeks-1:],
            sw_pred2w_death[i][nweeks-1:], 
            # sw_pred3w_death[i][nweeks-1:] 
        ]
        death_state_linechart(state, data, proj, pd3dates)


