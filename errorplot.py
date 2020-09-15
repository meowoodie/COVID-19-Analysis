import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from utils import table_loader, merge_counties_by_states

def error(real_data, pred_data, states, counties, c2s):
    error_mat = np.zeros((len(states) - 2, real_data.shape[1]))
    for i, is_county_belong_to in enumerate(c2s[:-2]):
        county_ind      = np.where(is_county_belong_to)[0]
        real            = real_data[county_ind]
        pred            = pred_data[county_ind]
        error           = (abs(real - pred)).mean(0)
        error_mat[i, :] = error

    state_real_confirm = merge_counties_by_states(real_data, states[:-2], counties, c2s[:-2]) # [ 50, nweeks ]
    state_pred_confirm = merge_counties_by_states(pred_data, states[:-2], counties, c2s[:-2]) # [ 50, nweeks ]
    error_state = (abs(state_real_confirm - state_pred_confirm)).mean(1)
    error_week  = (abs(state_real_confirm - state_pred_confirm)).mean(0)
    
    return error_mat, error_state, error_week

def error_heatmap(real_data, recv_data, states, counties, c2s, dates, nweeks, modelname, pred_data=None):

    dates = dates[:nweeks]

    error_mat, error_state, error_week    = error(real_data[:, :nweeks], recv_data[:, :nweeks], states, counties, c2s)
    if pred_data is not None:
        error_mat0, error_state0, error_week0 = error(real_data[:, :nweeks], pred_data[:, :nweeks], states, counties, c2s)
    error_mat1, error_state1, error_week1 = error(real_data[:, :nweeks-1], real_data[:, 1:nweeks], states, counties, c2s)
    print(error_mat.shape)
    print(len(dates))

    states_order = np.argsort(error_state)
    # states       = states[states_order]
    states       = [ states[ind] for ind in states_order ]
    error_mat    = error_mat[states_order, :]
    if pred_data is not None:
        error_mat0   = error_mat0[states_order, :]
    error_mat1   = error_mat1[states_order, :]
    rev_states_order = np.flip(states_order)
    error_state  = error_state[rev_states_order]
    if pred_data is not None:
        error_state0 = error_state0[rev_states_order]
    error_state1 = error_state1[rev_states_order]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 8}
    plt.rc('font', **font)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width       = 0.15, 0.65
    bottom, height    = 0.15, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_imshow = [left, bottom, width, height]
    rect_week   = [left, bottom_h, width, 0.12]
    rect_state  = [left_h, bottom, 0.12, height]

    with PdfPages("%s.pdf" % modelname) as pdf:
        # start with a rectangular Figure
        fig = plt.figure(1, figsize=(8, 8))

        ax_imshow = plt.axes(rect_imshow)
        ax_state  = plt.axes(rect_state)
        ax_week   = plt.axes(rect_week)

        # no labels
        ax_state.xaxis.set_major_formatter(nullfmt)
        ax_week.yaxis.set_major_formatter(nullfmt)

        # the error matrix for counties in states:
        cmap = matplotlib.cm.get_cmap('magma')
        img  = ax_imshow.imshow(np.log(error_mat + 1e-5), cmap=cmap, extent=[0,nweeks,0,50], aspect=float(nweeks)/50.)
        ax_imshow.set_yticks(np.arange(50))
        ax_imshow.set_yticklabels(states)
        ax_imshow.set_xticks(np.arange(nweeks))
        ax_imshow.set_xticklabels(dates, rotation=90)

        # the error vector for states and weeks
        ax_state.plot(np.log(error_state + 1e-5), np.arange(50), c="red", linewidth=2, linestyle="-", label="In-sample fitted", alpha=.5)
        if pred_data is not None:
            ax_state.plot(np.log(error_state0 + 1e-5), np.arange(50), c="blue", linewidth=2, linestyle="-", label="One-week ahead prediction", alpha=.5)
        ax_state.plot(np.log(error_state1 + 1e-5), np.arange(50), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

        ax_week.plot(np.log(error_week + 1e-5), c="red", linewidth=2, linestyle="-", label="In-sample fitted", alpha=.5)
        if pred_data is not None:
            ax_week.plot(np.log(error_week0 + 1e-5), c="blue", linewidth=2, linestyle="-", label="One-week ahead prediction", alpha=.5)
        ax_week.plot(np.log(error_week1 + 1e-5), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

        ax_state.get_yaxis().set_ticks([])
        ax_state.get_xaxis().set_ticks([])
        ax_state.set_xlabel("MAE")
        ax_state.set_ylim(0, 50)
        ax_week.get_xaxis().set_ticks([])
        ax_week.get_yaxis().set_ticks([])
        ax_week.set_ylabel("MAE")
        ax_week.set_xlim(0, nweeks)
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.91, 0.133, '%.2f' % max(max(error_state), max(error_state1)))
        plt.figtext(0.135, 0.81, '0')
        # plt.figtext(0.105, 0.915, '%.2f' % max(max(error_week), max(error_week1)))
        plt.figtext(0.095, 0.915, '%.2f' % max(max(error_week), max(error_week1)))
        plt.legend(loc="upper right")

        cbaxes = fig.add_axes([left_h, height + left + 0.01, .03, .12])
        cbaxes.get_xaxis().set_ticks([])
        cbaxes.get_yaxis().set_ticks([])
        cbaxes.patch.set_visible(False)
        cbar = fig.colorbar(img, cax=cbaxes)
        cbar.set_ticks([
            np.log(error_mat + 1e-5).min(), 
            np.log(error_mat + 1e-5).max()
        ])
        cbar.set_ticklabels([
            0, # "%.2e" % error_mat.min(), 
            "%.2f" % error_mat.max()
        ])
        cbar.ax.set_ylabel('MAE', rotation=270, labelpad=-5)

        fig.tight_layout()
        # plt.show()
        pdf.savefig(fig)

if __name__ == "__main__":

    # META DATA
    counties     = np.load("predictions/processed_data/counties.npy").tolist()
    states       = np.load("predictions/processed_data/states.npy").tolist()
    c2s          = np.load("predictions/processed_data/state_to_county.npy")

    # DATA
    real_death, real_dates = table_loader("predictions/deaths.csv", counties)
    pred_death, pred_dates = table_loader("predictions/predicted-deaths.csv", counties)
    recv_death, recv_dates = table_loader("predictions/recovered-deaths.csv", counties)

    real_death = real_death[:, 11:]
    recv_death = recv_death[:, 11:]
    real_dates = real_dates[11:]

    error_heatmap(real_death, recv_death, states, counties, c2s, real_dates, nweeks=len(real_dates)-1, modelname="death-error", pred_data=pred_death)



    # # META DATA
    # counties     = np.load("predictions/processed_data/counties.npy").tolist()
    # states       = np.load("predictions/processed_data/states.npy").tolist()
    # c2s          = np.load("predictions/processed_data/state_to_county.npy")

    # # DATA
    # real_case, real_dates = table_loader("predictions/cases.csv", counties)
    # recv_case, recv_dates = table_loader("predictions/recovered-cases.csv", counties)

    # # real_case = real_case[:, 11:]
    # # recv_case = recv_case[:, 11:]
    # # real_dates = real_dates[11:]

    # error_heatmap(real_case, recv_case, states, counties, c2s, real_dates, nweeks=len(real_dates)-1, modelname="case-error")