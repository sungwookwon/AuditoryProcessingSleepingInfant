import matplotlib.pyplot as plt
import numpy as np
import mne

############################################################################################################
### ----- Commented functions at the bottom are not relevant for analysis_mne_apice_optim notebook ----- ###
############################################################################################################


# --------- for delta summaries (scatter, histograms) ------------ #

def plot_delta(spindle_metrics):
    subjects = list(spindle_metrics.keys())
    n = len(subjects)

    fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))

    for i, subj in enumerate(subjects):
        delta = spindle_metrics[subj]["delta"]
        x = np.arange(len(delta))

        # Scatter
        axes[i, 0].scatter(x, delta, alpha=0.2)
        axes[i, 0].set_ylabel(f"{subj} delta")
        axes[i, 0].set_xlabel("Trials")

        # Histogram
        axes[i, 1].hist(delta, bins=30)
        axes[i, 1].set_ylabel("Count")
        axes[i, 1].set_xlabel("Delta")

        if i == 0:
            axes[i, 0].set_title("Delta across trials")
            axes[i, 1].set_title("Delta distribution")

    plt.tight_layout()
    plt.show()



# ---------- for a topomap of delta values. Useful for selecting ROIs for trial classification based on spindle/no spindle----------- #
# ---------- individual plots ----------- #

def plot_topo_delta(spindle_metrics, info):

    subjects = list(spindle_metrics.keys())
    n = len(subjects)

    delta = [spindle_metrics[subj]["delta"] for subj in subjects]

    n_rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    fig.suptitle("Delta power topomaps", fontsize=16)

    for i, subj in enumerate(subjects):

        d = delta[i]
        inf = info[subj]  

        vmin = d.min()
        vmax = d.max()

        img, _ = mne.viz.plot_topomap(
            d,
            inf,
            axes=axes[i],
            show=False,
            vlim=(vmin, vmax)
        )

        cbar = plt.colorbar(img, ax=axes[i], shrink=0.75, orientation='vertical')
        cbar.set_label('Delta/Electrode')

        axes[i].set_title(subj)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



### ----- Same as the one before, but average the delta values across subjects and plot it. 
### ----- This is to determine the subset of electrodes to calculate deltas on.
### ----- ! To use only with deltas averaged across subjects (124, ). check with "axis" parameter in compute_spindle_metric. 

def topo_delta_all(spindle_metrics, info):

    all_delta = [spindle_metrics[subj]["delta"] for subj in spindle_metrics]

    all_delta_zs = []

    for subj in all_delta:
        mean = np.mean(subj)
        std = np.std(subj)
        zscore_subj = []
        
        for delta in subj:
            zscore = (delta - mean)/std
            zscore_subj.append(zscore)
        
        all_delta_zs.append(zscore_subj)

    all_delta_zs_mean = np.mean(all_delta_zs, axis = 0)

    fig, ax = plt.subplots(1,1, figsize = (6, 6))

    img, _ = mne.viz.plot_topomap(
                all_delta_zs_mean,
                info["S28"],
                axes = ax,
                show=False
            )

    cbar = plt.colorbar(img, shrink=0.75, orientation='vertical')
    cbar.set_label('Delta/Electrode')
    ax.set_title("Topomap of delta values across subjects in z-score")



##----- PLOT FITS FOR PLAIN, SPINDLE, NO SPINDLE and NO SPINDLE IN SPINDLES -----##
##----- Results summary for compute_spindle_metrics()------##

def plot_fit(spindle_metrics, freqs, spindle_index, spindle_start, spindle_end):
    
    subjects = list(spindle_metrics.keys())
    n = len(subjects)
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(24, 4*n))

    col_titles = ["All trials", "Spindles", "No spindles in spindle cluster", "No spindles outside spindle cluster"]

    for i, subj in enumerate(subjects):
        signal, fit = spindle_metrics[subj]["log_psd"], spindle_metrics[subj]["fit"]
        spindles, no_spindles_in_spindles, valid_no_spindles = spindle_index[subj]["spindles"], spindle_index[subj]["ns_in_spindles"], spindle_index[subj]["valid_no_spindles"] 

        signal_all = signal.mean(axis=(0,1))
        signal_sp  = signal[spindles].mean(axis=(0,1))
        signal_ns  = signal[no_spindles_in_spindles].mean(axis=(0,1))
        signal_ns_sp = signal[valid_no_spindles].mean(axis=(0,1))

        fit_all = fit.mean(axis=(0,1))
        fit_sp  = fit[spindles].mean(axis=(0,1))
        fit_ns  = fit[no_spindles_in_spindles].mean(axis=(0,1))
        fit_ns_sp = fit[valid_no_spindles].mean(axis=(0,1))

        signals_cond = [signal_all, signal_sp, signal_ns, signal_ns_sp]
        fits_cond    = [fit_all, fit_sp, fit_ns, fit_ns_sp]

        for j in range(4):
            ax = axes[i, j]

            line = ax.plot(freqs, signals_cond[j], freqs, fits_cond[j])
            line[0].set_label("Original signal")
            line[1].set_label("Background")

            # Titles
            if i == 0:
                ax.set_title(col_titles[j])
            if j == 0:
                ax.set_ylabel(f"{subj}")

            if i == 0 and j == 0:
                ax.legend()

            ymin, ymax = ax.get_ylim() 
            ax.vlines(spindle_start, ymin=ymin, ymax=ymax, color = "r", linestyles='dashed', label = 'Spindle Band')
            ax.vlines(spindle_end, ymin=ymin, ymax=ymax, color = "r", linestyles='dashed')
            ax.axvspan(spindle_start, spindle_end, color='skyblue', alpha=0.2)

    fig.supxlabel("Frequency")
    fig.supylabel("Power(log)")

    plt.tight_layout()
    plt.show()


### -------- Plot kmeans results ------------- ######
### Classficiation & trajectory trace + Loss function & convergence

def plot_kmeans(result_kmeans, spindle_metrics, K):

    subjects = list(result_kmeans.keys())
    n = len(subjects)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 3))

    cluster_colors = np.array(["#d62728", "#0d36a5ea"])

    for i, subj in enumerate(subjects):

        data = result_kmeans[subj]
        labels = data["labels"]
        X = spindle_metrics[subj]["delta"]

        y_jitter = np.random.normal(0, 0.02, size=len(labels))

        # Scatter plot
        axes[i, 0].scatter(
            X,
            y_jitter,
            c=cluster_colors[labels],
            s=18,
            alpha=0.7
        )

        axes[i, 0].set_title(f"{subj} clustering")
        axes[i, 0].set_yticks([])

        # Cost function
        axes[i, 1].plot(data["cost_history"], marker='o')
        axes[i, 1].set_xlabel("Iteration")
        axes[i, 1].set_ylabel("Cost")
        axes[i, 1].set_title(f"Cost function over iterations - {subj}")
        axes[i, 1].grid()

        # Center trajectories
        centers_history = data["center_history"]

        for j in range(K):
            traj = centers_history[:, j, 0]
            y_traj = np.zeros_like(traj)

            axes[i, 0].plot(
                traj,
                y_traj,
                marker='o',
                linestyle='-',
                color='black',
                label=f"Center {j}" if i == 0 else None
            )

            # Start center
            axes[i, 0].scatter(
                traj[0],
                0,
                marker='s',
                s=90,
                edgecolor='k',
                facecolor='black'
            )

            # End center
            axes[i, 0].scatter(
                traj[-1],
                0,
                marker='X',
                s=120,
                edgecolor='k',
                facecolor='black'
            )

        axes[i, 0].set_title(f"{subj} center trajectories")
        axes[i, 0].set_yticks([])

    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()



### plot the results of window sliding with information about spindles

def plot_sp_ratio_window(spindle_window, spindle_index):

    fig, axes = plt.subplots(9, 3, figsize = (18, 45))
    axes = axes.flatten()
    
    mean_s, std_s = np.mean([len(v["spindles"]) for v in spindle_index.values()]), np.std([len(v["spindles"]) for v in spindle_index.values()])
    mean_ns, std_ns = np.mean([len(v["ns_in_spindles"]) for v in spindle_index.values()]), np.std([len(v["ns_in_spindles"]) for v in spindle_index.values()])
    conc_gb = mean_s/(mean_ns + mean_s)

    for i, subj in enumerate(spindle_window):

        num_s =  len(spindle_index[subj]["spindles"])
        num_ns = len(spindle_index[subj]["ns_in_spindles"])
        conc =  num_s/(num_ns+num_s)

        data = spindle_window[subj]

        textstr0 = f"spindle counts = {num_s:.2f}\n non spindle counts= {num_ns:.2f} \n mean spindle = {mean_s:.2f} \n mean none spindles = {mean_ns:.2f} \n mean sp concentration = {conc_gb:.2f}% "
        textstr  = f"spindle counts = {num_s:.2f}\n non spindle counts= {num_ns:.2f}\n spindle concentration = {conc:.2f}%"

        axes[i].scatter(np.arange(len(data["x"])), data["y"], alpha = 0.3, color="purple")
        axes[i].set(title = f"{subj}",
                    ylabel = "ratio spindles over 50 trial window",
                    xlabel = "steps")
        if i == 0:
            axes[i].text(
                0.95, 0.95, textstr0,
                transform=axes[i].transAxes,   # relative coords (0–1)
                fontsize=10,
                #color = 'r',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
        else : 
            axes[i].text(
                0.95, 0.95, textstr,
                transform=axes[i].transAxes,   # relative coords (0–1)
                fontsize=10,
                #color = 'r',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
        

### plot frequency associated with maximum value among spindle trials. (method = subtraction between log_psd and fit)

def plot_freq_maxval_sp(spindle_metrics, spindle_index, method, freqs, mask_spindle):

    if method not in ["individual", "group"]:
        raise ValueError("Please choose between 'individual' and 'group'.")

    if method == "individual":

        fig, axes = plt.subplots(9, 3 , figsize = (18, 45))
        axes = axes.flatten()

        for i, subj in enumerate(spindle_metrics.keys()):
            sp = spindle_index[subj]["spindles"]
            data, fit = spindle_metrics[subj]["log_psd"][sp].mean(axis = 1), spindle_metrics[subj]["fit"].mean(axis = 1)
            max_frq = []
            for trial, fit in zip(data, fit):
                max_idx = np.argmax(trial[mask_spindle] - fit[mask_spindle])
                frq = freqs[mask_spindle][max_idx]
                max_frq.append(frq)
            axes[i].hist(max_frq, bins = freqs[mask_spindle], alpha=0.7, edgecolor='black', linewidth=0.8)
            axes[i].set_title(f"{subj}")
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel("Count")
        
    elif method == "group":

        max_sp_frq = []

        for subj in spindle_metrics.keys():
            data = spindle_metrics[subj]["log_psd"]
            fit = spindle_metrics[subj]["fit"]
            sp = spindle_index[subj]["spindles"]
            max_idx = np.argmax(data[sp].mean(axis = (0,1))[mask_spindle] - fit.mean(axis = (0, 1))[mask_spindle])
            max_sp_frq.append(freqs[mask_spindle][max_idx])

        plt.hist( max_sp_frq, bins = 10, alpha=0.7, edgecolor='black', linewidth=0.8)
        plt.title("Distribution of max values per frequency in spindle band (group level)")
        plt.xlabel("Frequency")
        plt.ylabel("Count(Nb of subjects)")

###----- plot counts of successive spindles -----###

# def plot_spcount(spcount):

#     medians = []

#     for subj in spcount.keys():
#         data = spcount[subj]["spindle_count"]
#         medians.append(np.median(data))

#     global_median = np.mean(medians)

#     n = len(spcount)

#     n_cols = 3
#     n_rows = int(np.ceil(n / n_cols))

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), sharex=True, sharey=True)
#     axes = axes.flatten()

#     median_list = []

#     for i, subj in enumerate(spcount.keys()):
#         data = spcount[subj]["spindle_count"]
#         axes[i].hist(data)
#         axes[i].set_xlabel("Number of successive spindles")
#         axes[i].set_ylabel("Occurrence")
#         axes[i].set_title(f'{subj}')
#         axes[i].set_xlim(1,10)
#         # axes[i].set_xticks(np.arange(len()))
#         median = np.median(data)
#         median_list.append(median)
#         mean = np.mean(data)

#         textstr = f"mean = {mean:.2f}\nmedian = {median:.2f} \n Mean of medians = {global_median:.2f}"

#         # --- add box ---
#         axes[i].text(
#             0.95, 0.95, textstr,
#             transform=axes[i].transAxes,   # relative coords (0–1)
#             fontsize=10,
#             #color = 'r',
#             verticalalignment='top',
#             horizontalalignment='right',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
#         )

#         # --- add vline ---

#         axes[i].axvline(median,  ls = "-", color='r', label = "Median")
#         axes[i].axvline(global_median, ls = "-", color = 'g', label = "Mean of medians")
#         axes[i].legend()


        
#     plt.tight_layout()
#     plt.show()



###------- plot topomap throughout successive timepoints. Useful for identifying ERP ROIs

# def plot_topo_per_cond(epochs, method, cond = None, spindle_index = None, times = [-0.2, 0.1, 0.4, 0.6, 1.0]):

#     if method not in ["individual", "average"]:
#         raise ValueError("Method should be either 'individual' or 'average'.")
    
#     if method == "individual":
    
#         for subj in epochs: 
#             epochs[subj].average().plot_topomap(times = times, show = False)

#     elif method == "average":

#         if cond is None or spindle_index is None:
#             raise ValueError(
#                 "For method='average', both cond and spindle_index are required."
#             )
        
#         cond_map = {
#         "spindles": "spindles",
#         "no spindles in spindles": "ns_in_spindles",
#         "valid no spindles": "no_spindles"
#     }

#         if cond not in cond_map:
#             raise ValueError(
#                 "Condition should be 'spindles', "
#                 "'no spindles in spindles', "
#                 "or 'valid no spindles'."
#             )

#         av_list = []
#         key = cond_map[cond] 

#         for subj in epochs.keys():
            
#             data =  epochs[subj]
#             trials = spindle_index[subj][key]

#             av_list.append(data[trials].average())

#         av = mne.grand_average(av_list)
#         av.plot_topomap(times = times, show = False, vlim = (-12, 12))


### plot temporal distribution of delta with different colors for 3 conditions + the temporal evolution of the
### ERP peak or background signal slope.

# def plot_delta_cond(epochs, spindle_metrics, spindle_index, effect="None" ):

#     fig, axes = plt.subplots(9, 3, figsize = (15, 45))
#     ax = axes.flatten()

#     epochs_peak = epochs.copy()

#     for i, subj in enumerate(epochs_peak):

#         peak = spindle_metrics[subj]["peak"]
#         delta = spindle_metrics[subj]["delta"]
#         coef = spindle_metrics[subj]["coef_mat"][:, :, 0].mean(axis=(1))
        

#         s = spindle_index[subj]["spindles"]
#         ns = spindle_index[subj]["no_spindles"]
#         nsis = spindle_index[subj]["ns_in_spindles"]

#         x = np.arange(len(delta))

#         # Scatter
#         ax[i].scatter(x[s], delta[s], alpha=0.3, color = "#d62728", label = 'spindle')
#         ax[i].scatter(x[ns], delta[ns], alpha=0.3, color = "#0d36a5ea", label = 'no spindle')
#         ax[i].scatter(x[nsis], delta[nsis], alpha=0.3, color = "#0da530ac", label = "no spindle in spindle")
#         ax[i].set_ylabel("Delta value range")
#         ax[i].set_xlabel("Trials")

#         if effect is not None:
#             ax2 = ax[i].twinx()

#             if effect == "peak":
#                 ax2.plot(x, peak, color="black", label="peak amplitude")

#             elif effect == "slope":
#                 ax2.plot(x, coef, color="black", label="peak (voltage)")
#                 ax2.set_ylabel("Peak voltage")

#         # Combine legends (important)
#         lines1, labels1 = ax[i].get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax[i].legend(lines1 + lines2, labels1 + labels2, loc="best")

#         ax[i].set_title(f"Temporal distribution of delta values for {subj}")
#         ax[i].legend()

#     plt.tight_layout()
#     plt.show()



### ------- compare ERPs between spindle vs no_spindles trials --------- ###

# def compare_erp_ind(epochs, spindle_index, picks):  

#         rng = np.random.default_rng()

#         n_rows = int(np.ceil(len(epochs)/ 2))
#         fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
#         axes = axes.flatten()

#         for i, subj in enumerate(epochs.keys()):
#             epoch = epochs[subj]
#             spindles, ns_in_spindles, no_spindles = spindle_index[subj]["spindles"], spindle_index[subj]["ns_in_spindles"], spindle_index[subj]["valid_no_spindles"]


#             evokeds = {"spindle" : epoch[spindles].average(),
#                         "no_spindle" : epoch[no_spindles].average(),
#                         "no_sp_in_sp" : epoch[ns_in_spindles].average()}

#             colors = plt.cm.viridis(np.linspace(0, 1, 3))
#             colors = [tuple(c) for c in colors]

#             mne.viz.plot_compare_evokeds(evokeds, picks, axes=axes[i], combine="mean", colors=colors, legend=True, title= f"Comparison sp vs no_sp vs no_sp_in_sp - {subj}", show=False)