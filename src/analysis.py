import numpy as np
import matplotlib.pyplot as plt
import mne
rng = np.random.default_rng()

# set electrodes to use

def set_electrodes(elecs_interest_sp, elecs_interest_erp, info):

    selected_elecs_sp = sorted([f"E{s}" for s in elecs_interest_sp])
    selected_elecs_erp = sorted([f"E{s}" for s in elecs_interest_erp])
        
    picks_sp = mne.pick_channels(info['ch_names'], include=selected_elecs_sp)
    picks_erp = mne.pick_channels(info['ch_names'], include=selected_elecs_erp)
    picks_all = mne.pick_channels(info["ch_names"], include=[])

    print(f"The chosen electrodes for spindle selection are {sorted(picks_sp)}.")
    print(f"The chosen electrodes for the ERP are {sorted(picks_erp)}.")
    print(f"All the electrodes are {sorted(picks_all)}.")


    return picks_sp, picks_erp, picks_all 

# PSD

def psd(epochs, *, fmin, fmax, picks=None):
    psd = epochs.compute_psd(method = "welch", fmin = fmin, fmax = fmax, picks = picks)
    return psd


# delta calculation 

def compute_spindle_metric(psd, log_freqs, mask_fit, mask_spindle, *, axis):

    log_psd = np.log10(psd) 
    background_fit = np.zeros_like(log_psd)
    
    n_epochs = log_psd.shape[0]
    n_chs = log_psd.shape[1]
    coef_mat = np.zeros((n_epochs, n_chs, 2))

    # fit 1/f background excluding spindle + harmonic bands
    for epoch in range(log_psd.shape[0]):
        for ch in range(log_psd.shape[1]):
            y = log_psd[epoch, ch, mask_fit]
            x = log_freqs[mask_fit]

            coef = np.polyfit(x, y, 1)
            coef_mat[epoch, ch, :] = coef

            background_fit[epoch, ch, :] = np.polyval(coef, log_freqs)

    # residual spectrum
    residual = log_psd - background_fit

  
     # integrate spindle band
    spindle_area = np.trapezoid(
        residual[:, :, mask_spindle],
        log_freqs[mask_spindle],
        axis=2
        ).mean(axis=axis) # change this axis to 1 if to average over channels and to 0 over trials.


    return log_psd, background_fit, residual, spindle_area, coef_mat 



### --------- Successive spindle count ------------- ###

def spindle_window(spindles_index):

    spcount = {}

    for subj in spindles_index:

        spindles = spindles_index[subj]["spindles"]

        spindle_count = []
        count = 1   # start at 1 (first element in a run)

        for i in range(1, len(spindles)):

            if spindles[i] == spindles[i-1] + 1:
                count += 1
            else:
                spindle_count.append(count)
                count = 1   # reset for new run

        # append last run
        if len(spindles) > 0:
            spindle_count.append(count)

        spcount[subj] = {"spindle_count": spindle_count}

    return spcount

### slide a window of length *window* by steps of 10 over the delta values of the trials.
### output :  returns a dictionary where subj : ratio of spindles trials over each window.
### use plot_sp_ratio_window to plot the results.

def window_delta(spindle_metrics, spindle_index, window):

    spindle_window = {}

    for subj in spindle_metrics.keys():

        data = spindle_metrics[subj]["delta"]
        spindles = spindle_index[subj]["spindles"]

        starts = []
        counts = []

        for i in range(0, len(data) - window +1, 10):
            count = np.sum((spindles >= i) & (spindles < i + window))
            counts.append(count / window)
            starts.append(i)

        spindle_window[subj] = {
            "x": np.array(starts),
            "y": np.array(counts)
        }

    return spindle_window     

### from the spindle_window output by window_delta(), extract the indices of none spindle trials that are among spindle trials.
### output :  dictionary where subj : indices of none spindles trials among spindle trials.
### to use in compare_erp()

def extract_ns(spindle_window, spindle_index, threshold, window : int):

    for subj in spindle_window : 
        
        x = spindle_window[subj]["x"]
        y = spindle_window[subj]["y"]
        ns = spindle_index[subj]["no_spindles"]

        selected_ns = []

        for start, density in zip(x, y):

            if density > threshold:
                window_trials = np.arange(start, start + window)
                trials_ns = np.intersect1d(window_trials, ns)
                selected_ns.extend(trials_ns)
            
        spindle_index[subj]["ns_in_spindles"] = np.unique(selected_ns)
        spindle_index[subj]["spindle_episode"] = np.concatenate((np.unique(selected_ns), spindle_index[subj]["spindles"]))

    return spindle_index

# Plot ERP and prepare matrix for permutation test

def prep_permute_and_erp_optim(list_s, list_ns, list_ns_in_s, picks):
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    X_spindle = np.array([
    evoked.data.T
    for evoked in list_s
    ])

    X_no_spindle = np.array([
        evoked.data.T
        for evoked in list_ns
    ])

    X_ns = np.array([
        evoked.data.T
        for evoked in list_ns_in_s
    ])

    X = [X_spindle, X_no_spindle, X_ns]
            
    evokeds = {
                    "spindle": list_s,
                    "no_spindle": list_ns,
                    "no_spindle in spindles" : list_ns_in_s
                                                    }

    colors = {
            "spindle": "#d62728",   
            "no_spindle": "#0d36a5ea", 
            "no_spindle in spindles" : "#0da530ac"
            }


    mne.viz.plot_compare_evokeds(evokeds, axes=axes, picks=picks, combine = "mean", colors=colors, legend=True, title= f"Comparison sp vs no_sp vs no_sp_in_sp - Grand average", show=False)
    return X, list_s, list_ns, list_ns_in_s, evokeds


# def prep_permute_and_erp(epochs, spindle_index, picks):
    
#     fig, axes = plt.subplots(1, 1, figsize=(6, 6))
                
#     list_s = []
#     list_ns = []
#     list_ns_in_s = []
#     evokeds = {}

#     for subj in epochs.keys():
            
#             epoch = epochs[subj]
            
#             if subj == "S34":
#                 valid_trials = np.arange(200, len(epoch))
            
#             else :
#                 valid_trials = np.where(
#                     epoch.metadata["sourdine"].values >= 0.2
#                 )[0]

#             spindles = np.intersect1d(
#                 spindle_index[subj]["spindles"],
#                 valid_trials
#             )

#             no_spindles = np.intersect1d(
#                 spindle_index[subj]["no_spindles"],
#                 valid_trials
#             )

#             ns_in_spindles = np.intersect1d(
#                 spindle_index[subj]["ns_in_spindles"],
#                 valid_trials
#             )
#             # spindles, no_spindles, ns_in_spindles = spindle_index[subj]["spindles"], spindle_index[subj]["no_spindles"], spindle_index[subj]["ns"]
#             list_s.append(epoch[spindles].average())
#             list_ns_in_s.append(epoch[ns_in_spindles].average())
#             valid_no_spindles = np.setdiff1d(
#                                                 no_spindles,
#                                                 ns_in_spindles
#                                             )
#             spindle_index[subj]["valid_no_spindles"] = valid_no_spindles
#             #subj_list_ns = []

#             if len(valid_no_spindles) <= len(spindles):
#                  samples = valid_no_spindles
#             # replace_flag = len(valid_no_spindles) < len(spindles)
#             else : 
#                  samples = rng.choice(valid_no_spindles, size=len(spindles), replace=False)
#             evoked_ns = epoch[samples].average()
#             #subj_list_ns.append(evoked_ns)
#             #subj_list_ns_av = mne.grand_average(subj_list_ns)
#             list_ns.append(evoked_ns) 
#             ### so that each subject contributes equally, not biased in terms of the number of trials of each subject.

#     X_spindle = np.array([
#     evoked.data.T
#     for evoked in list_s
#     ])

#     X_no_spindle = np.array([
#         evoked.data.T
#         for evoked in list_ns
#     ])

#     X_ns = np.array([
#         evoked.data.T
#         for evoked in list_ns_in_s
#     ])

#     X = [X_spindle, X_no_spindle, X_ns]
            
    
#     # dict_ns = {"no_spindle" : mne.grand_average(list_ns)}
#     # dict_s = {"spindle" : mne.grand_average(list_s)}

#     evokeds = {
#                     "spindle": list_s,
#                     "no_spindle": list_ns,
#                     "no_spindle in spindles" : list_ns_in_s
#                                                     }

#     colors = {
#             "spindle": "#d62728",   
#             "no_spindle": "#0d36a5ea", 
#             "no_spindle in spindles" : "#0da530ac"
#             }


#     mne.viz.plot_compare_evokeds(evokeds, axes=axes, picks=picks, combine = "mean", colors=colors, legend=True, title= f"Comparison sp vs no_sp vs no_sp_in_sp - Grand average", show=False)
#     return X, list_s, list_ns, list_ns_in_s, spindle_index, evokeds

