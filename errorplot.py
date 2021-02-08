import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from utils import table_loader, merge_counties_by_states, statewise_baselines_loader

def error(real_data, pred_data, states, counties, c2s):
    error_mat = np.zeros((len(states) - 2, real_data.shape[0]))
    for i, is_county_belong_to in enumerate(c2s[:-2]):
        county_ind      = np.where(is_county_belong_to)[0]
        real            = real_data[:, county_ind]
        pred            = pred_data[:, county_ind]
        error           = (abs(real - pred)).mean(1)
        error_mat[i, :] = error

    state_real_confirm = merge_counties_by_states(real_data, states[:-2], counties, c2s[:-2]) # [ nweeks, 50 ]
    state_pred_confirm = merge_counties_by_states(pred_data, states[:-2], counties, c2s[:-2]) # [ nweeks, 50 ]
    error_state = (abs(state_real_confirm - state_pred_confirm)).mean(0)                      # [ 50 ]
    error_week  = (abs(state_real_confirm - state_pred_confirm)).mean(1)                      # [ nweeks ]

    return error_mat, error_state, error_week

def est_error_heatmap(data, states, counties, c2s, dates, nweeks, modelname):

    dates = dates[-nweeks:]

    real_data      = data[0]
    est_data       = data[1]
    est_data_nocov = data[2]
    est_data_nomob = data[3]

    error_mat, error_state, error_week    = error(real_data[-nweeks:, :], est_data[-nweeks-1:-1, :], states, counties, c2s)
    error_mat1, error_state1, error_week1 = error(real_data[-nweeks:, :], est_data_nomob[-nweeks-1:-1, :], states, counties, c2s)
    error_mat2, error_state2, error_week2 = error(real_data[-nweeks:, :], est_data_nocov[-nweeks-1:-1, :], states, counties, c2s)
    print(error_mat.shape)
    print(len(dates))

    states_order = np.argsort(error_state)
    states       = [ states[ind] for ind in states_order ]
    error_mat    = error_mat[states_order, :]
    # error_mat1   = error_mat1[states_order, :]
    # error_mat2   = error_mat2[states_order, :]
    rev_states_order = np.flip(states_order)
    error_state  = error_state[rev_states_order]
    error_state1 = error_state1[rev_states_order]
    error_state2 = error_state2[rev_states_order]

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
        ax_state.plot(np.log(error_state + 1e-5), np.arange(50), c="red", linewidth=2, linestyle="-", label="STVA", alpha=.5)
        ax_state.plot(np.log(error_state1 + 1e-5), np.arange(50), c="blue", linewidth=1.5, linestyle="--", label=r"STVA$-$mobility", alpha=.5)
        ax_state.plot(np.log(error_state2 + 1e-5), np.arange(50), c="green", linewidth=1.5, linestyle="--", label=r"STVA$-$census", alpha=.5)

        ax_week.plot(np.log(error_week + 1e-5), c="red", linewidth=2, linestyle="-", label="STVA", alpha=.5)
        ax_week.plot(np.log(error_week1 + 1e-5), c="blue", linewidth=1.5, linestyle="--", label=r"STVA$-$mobility", alpha=.5)
        ax_week.plot(np.log(error_week2 + 1e-5), c="green", linewidth=1.5, linestyle="--", label=r"STVA$-$census", alpha=.5)

        ax_state.get_yaxis().set_ticks([])
        ax_state.get_xaxis().set_ticks([])
        ax_state.set_xlabel("Avg MAE")
        ax_state.set_ylim(0, 50)
        ax_week.get_xaxis().set_ticks([])
        ax_week.get_yaxis().set_ticks([])
        ax_week.set_ylabel("Avg MAE")
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

def pred_error_heatmap(data, states, counties, c2s, dates, nweeks, modelname):

    def statewise_error(real_data, pred_data, states, counties, c2s):
        error_state, error_week = np.zeros(50), np.zeros(real_data.shape[0])
        real_data = merge_counties_by_states(real_data, states[:-2], counties, c2s[:-2])
        for i, state in enumerate(states[:-2]):
            real            = real_data[:, i]
            pred            = pred_data[:, i]
            error           = (abs(real - pred)).mean()
            error_state[i]  = error
        
        for t in range(real_data.shape[0]):
            real            = real_data[t, :]
            pred            = pred_data[t, :-2]
            error           = (abs(real - pred)).mean()
            error_week[t]  = error

        return None, error_state, error_week

    dates = dates[-nweeks:]

    real_data = data[0]
    pred_data = data[1]
    ut_data   = data[2]
    lanl_data = data[3]
    mobs_data = data[4]
    ihme_data = data[5]

    error_mat, error_state, error_week = error(real_data[-nweeks:, :], pred_data[-nweeks:, :], states, counties, c2s)
    _, error_state1, error_week1       = statewise_error(real_data[-nweeks:, :], ut_data[-nweeks-1:-1, :], states, counties, c2s)
    _, error_state2, error_week2       = statewise_error(real_data[-nweeks:, :], lanl_data[-nweeks-1:-1, :], states, counties, c2s)
    _, error_state3, error_week3       = statewise_error(real_data[-nweeks:, :], mobs_data[-nweeks-1:-1, :], states, counties, c2s)
    _, error_state4, error_week4       = statewise_error(real_data[-nweeks:, :], ihme_data[-nweeks-1:-1, :], states, counties, c2s)
    print(error_mat.shape)
    print(len(dates))

    states_order = np.argsort(error_state)
    states       = [ states[ind] for ind in states_order ]
    error_mat    = error_mat[states_order, :]
    rev_states_order = np.flip(states_order)
    error_state  = error_state[rev_states_order]
    error_state1 = error_state1[rev_states_order]
    error_state2 = error_state2[rev_states_order]
    error_state3 = error_state3[rev_states_order]
    error_state4 = error_state4[rev_states_order]

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
        ax_state.plot(np.log(error_state + 1e-5), np.arange(50), c="red", linewidth=2, linestyle="-", label="STVA", alpha=.5)
        ax_state.plot(np.log(error_state1 + 1e-5), np.arange(50), c="blue", linewidth=1.5, linestyle="--", label="UT", alpha=.5)
        ax_state.plot(np.log(error_state2 + 1e-5), np.arange(50), c="green", linewidth=1.5, linestyle="--", label="LANL", alpha=.5)
        ax_state.plot(np.log(error_state3 + 1e-5), np.arange(50), c="purple", linewidth=1.5, linestyle="--", label="MOBS", alpha=.5)
        ax_state.plot(np.log(error_state4 + 1e-5), np.arange(50), c="orange", linewidth=1.5, linestyle="--", label="IHME", alpha=.5)

        ax_week.plot(np.log(error_week + 1e-5), c="red", linewidth=2, linestyle="-", label="STVA", alpha=.5)
        ax_week.plot(np.log(error_week1 + 1e-5), c="blue", linewidth=1.5, linestyle="--", label="UT", alpha=.5)
        ax_week.plot(np.log(error_week2 + 1e-5), c="green", linewidth=1.5, linestyle="--", label="LANL", alpha=.5)
        ax_week.plot(np.log(error_week3 + 1e-5), c="purple", linewidth=1.5, linestyle="--", label="MOBS", alpha=.5)
        ax_week.plot(np.log(error_week4 + 1e-5), c="orange", linewidth=1.5, linestyle="--", label="IHME", alpha=.5)

        ax_state.get_yaxis().set_ticks([])
        ax_state.get_xaxis().set_ticks([])
        ax_state.set_xlabel("Avg MAE")
        ax_state.set_ylim(0, 50)
        ax_week.get_xaxis().set_ticks([])
        ax_week.get_yaxis().set_ticks([])
        ax_week.set_ylabel("Avg MAE")
        ax_week.set_xlim(0, nweeks)
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.91, 0.133, '%.2f' % max(max(error_state), max(error_state1)))
        plt.figtext(0.135, 0.81, '0')
        # plt.figtext(0.105, 0.915, '%.2f' % max(max(error_week), max(error_week1)))
        plt.figtext(0.095, 0.915, '%.2f' % max(max(error_week), max(error_week1)))
        plt.legend(loc="upper left")

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
    counties   = np.load("mat/counties.npy").tolist()
    states     = np.load("mat/states.npy").tolist()
    c2s        = np.load("mat/state_to_county.npy")
    real_dates = np.load("mat/weeks_1-17.npy")           # [ n_weeks ]

    # IN-SAMPLE ESTIMATION 
    real_death      = np.load("mat/death_1-17.npy")      # [ n_weeks, n_counties ]
    pred_death      = np.load("mat/onestep/onestep.npy") # [ 15, n_counties ]
    est_death       = np.load("mat/insample/model.npy")  
    est_death_nocov = np.load("mat/insample/no_cov.npy")
    est_death_nomob = np.load("mat/insample/no_mob.npy")
    print(pred_death.shape)

    # make negative values zero
    est_death[est_death<0]             = 0
    est_death_nocov[est_death_nocov<0] = 0
    est_death_nomob[est_death_nomob<0] = 0

    # data = [ real_death, est_death, est_death_nocov, est_death_nomob ]
    # est_error_heatmap(data, states, counties, c2s, real_dates, nweeks=45, modelname="est-death-error")

    # OUT-OF-SAMPLE PREDICTION
    preds = statewise_baselines_loader()
    data  = [ real_death, pred_death, preds["UT"], preds["LANL"], preds["MOBS"], preds["IHME"] ]
    pred_error_heatmap(data, states, counties, c2s, real_dates, nweeks=15, modelname="pred-death-error")