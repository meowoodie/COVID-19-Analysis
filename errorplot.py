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
        error           = ((real - pred) ** 2).mean(0)
        error_mat[i, :] = error

    sw_real_confirm = merge_counties_by_states(real_data, states[:-2], counties, c2s[:-2]) # [ 50, nweeks ]
    sw_pred_confirm = merge_counties_by_states(pred_data, states[:-2], counties, c2s[:-2]) # [ 50, nweeks ]
    error_state = (abs(sw_real_confirm - sw_pred_confirm)).mean(1)
    error_week  = (abs(sw_real_confirm - sw_pred_confirm)).mean(0)
    
    return error_mat, error_state, error_week

def error_heatmap(real_data, pred_data, states, counties, c2s, dates, nweeks=22, modelname="One-week ahead"):

    dates = dates[:nweeks]

    error_mat, error_state, error_week    = error(real_data[:, :nweeks], pred_data[:, :nweeks], states, counties, c2s)
    error_mat0, error_state0, error_week0 = error(real_data[:, 1:nweeks+1], real_data[:, :nweeks], states, counties, c2s)
    print(error_mat.shape)
    print(len(dates))

    states_order = np.argsort(error_state)
    # states       = states[states_order]
    states       = [ states[ind] for ind in states_order ]
    error_mat    = error_mat[states_order, :]
    error_mat0   = error_mat0[states_order, :]
    rev_states_order = np.flip(states_order)
    error_state  = error_state[rev_states_order]
    error_state0 = error_state0[rev_states_order]

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
        ax_state.plot(np.log(error_state + 1e-5), np.arange(50), c="red", linewidth=2, linestyle="-", label="%s" % modelname, alpha=.8)
        ax_state.plot(np.log(error_state0 + 1e-5), np.arange(50), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)
        ax_week.plot(np.log(error_week + 1e-5), c="red", linewidth=2, linestyle="-", label="%s" % modelname, alpha=.8)
        ax_week.plot(np.log(error_week0 + 1e-5), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

        ax_state.get_yaxis().set_ticks([])
        ax_state.get_xaxis().set_ticks([])
        ax_state.set_xlabel("MAE")
        ax_state.set_ylim(0, 50)
        # plt.setp(ax_state.xaxis.get_label(), visible=True, text="MSE")
        ax_week.get_xaxis().set_ticks([])
        ax_week.get_yaxis().set_ticks([])
        ax_week.set_ylabel("MAE")
        ax_week.set_xlim(0, nweeks)
        # plt.setp(ax_week.yaxis.get_label(), visible=True, text="MSE")
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.91, 0.133, '%.2e' % max(max(error_state), max(error_state0)))
        plt.figtext(0.135, 0.81, '0')
        plt.figtext(0.085, 0.915, '%.2e' % max(max(error_week), max(error_week0)))
        plt.legend()

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
            "%.2e" % error_mat.max()
        ])
        cbar.ax.set_ylabel('MAE', rotation=270, labelpad=-20)

        fig.tight_layout()
        # plt.show()
        pdf.savefig(fig)

if __name__ == "__main__":

    # META DATA
    counties     = np.load("predictions/processed_data/counties.npy").tolist()
    states       = np.load("predictions/processed_data/states.npy").tolist()
    c2s          = np.load("predictions/processed_data/state_to_county.npy")

    # DATA
    real_confirm, rcdates = table_loader("predictions/confirmedCases.csv", counties)
    real_death, rddates   = table_loader("predictions/Deaths.csv", counties)

    pred_confirm_1week, pc1dates = table_loader("predictions/oneWeekCasesPrediction.csv", counties)
    pred_confirm_2week, pc2dates = table_loader("predictions/twoWeekCasesPrediction.csv", counties)
    pred_death_1week, pd1dates   = table_loader("predictions/oneWeekDeathPrediction.csv", counties)
    pred_death_2week, pd2dates   = table_loader("predictions/twoWeekDeathPrediction.csv", counties)
    pred_death_3week, pd3dates   = table_loader("predictions/threeWeekDeathPrediction.csv", counties)

    # error_heatmap(real_confirm, pred_confirm_1week, states, counties, c2s, rcdates, nweeks=24, modelname="1-week ahead confirm")
    # error_heatmap(real_confirm, pred_confirm_2week, states, counties, c2s, rcdates, nweeks=21, modelname="2-weeks ahead confirm")
    error_heatmap(real_death, pred_death_1week, states, counties, c2s, rcdates, nweeks=24, modelname="1-week ahead death")
    # error_heatmap(real_death, pred_death_2week, states, counties, c2s, rcdates, nweeks=21, modelname="2-weeks ahead death")
    # error_heatmap(real_death, pred_death_3week, states, counties, c2s, rcdates, nweeks=20, modelname="3-weeks ahead death")


    # cmap = matplotlib.cm.get_cmap('magma')
    # plt.imshow(np.log(error_mat + 1e-5), cmap=cmap, extent=[0,23,0,50], aspect=.5)
    # plt.show()

    # plt.plot(np.log(error_state + 1e-5))
    # plt.plot(np.log(error_state0 + 1e-5))
    # plt.show()

    # plt.plot(np.log(error_week + 1e-5))
    # plt.plot(np.log(error_week0 + 1e-5))
    # plt.show()