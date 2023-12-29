import scipy.signal as ss
from glob import glob
import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter
import icsd
import quantities as pq
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_mua(ecp, filter_order = 5, fs = 20000, fc = 500, q = 20):
    '''
        This function gives you the MUA from the ECP
        Parameters
        ---------
                ecp : extracellular potential
                filter_order : order of butterworth filter
                fs : sampling frequency (Hz)
                fc : cut-off frequency
                q : downsampling order
        Returns
        ---------
                mua : multi-unit activity
    '''
    # creating high-pass filter
    Wn = fc/fs/2
    
    b, a = ss.butter(filter_order, Wn, btype = 'highpass')
    
    mua = ss.filtfilt(b, a, ecp, axis = 0)

    # downsample to 1 kHz
    for q_ in [10, q // 10]:
        mua = ss.decimate(mua, q_, axis = 0)
    
    mua = abs(mua)
    
    return mua


def get_lfp(ecp, filter_order = 5, fs = 10000, fc = 500, q = 20, downsample = True, do_filter = True):
    '''
        This function gives you the LFP from the ECP
        Parameters
        ---------
                ecp : extracellular potential
                filter_order : order of butterworth filter
                fs : sampling frequency (Hz)
                fc : cut-off frequency
                q : downsampling order
        Returns
        ---------
                lfp : local field potential
    '''
    if do_filter:
        # creating high-pass filter
        Wn = fc/fs/2

        b, a = ss.butter(filter_order, Wn, btype = 'low')

        lfp = ss.filtfilt(b, a, ecp, axis = 0)
    else:
        lfp = ecp

    if downsample:
        for q_ in [10, q // 10]:
            lfp = ss.decimate(lfp, q_, axis = 0)
    
    return lfp

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_stimulus_lfp(session, lfp, presentation_nr_start = 0, presentation_nr_stop = -1,\
                    time_before_stim_onset = 0.0, time_after_stim_offset = 0.0, stim_name = 'flashes'):
    '''
    This function extracts the experimental LFP during the presentation of a selected stimulus.
    Parameters
    ---------
            session : loaded nwb file with session data
            lfp : lfp from whole period
            presentation_nr_start : which stimulus presentation to start at
            presentation_nr_start : which stimulus presentation to stop at
            time_before_stim_onset : time before stimulus onset to include in each trial
            time_after_stim_onset : time after stimulus onset to include in each trial
            stim_name : stimulus key name
        Returns
        ---------
            lfp : local field potential
    '''
    
    stimulus_presentation = session.stimulus_presentations[
        session.stimulus_presentations['stimulus_name'] == stim_name
    ]
    
    start_time_stim = stimulus_presentation['start_time'].values[presentation_nr_start]+time_before_stim_onset
    
    stop_time_stim = stimulus_presentation['stop_time'].values[presentation_nr_stop]+time_after_stim_offset
    
    idx_start_stim = find_nearest(lfp.time.values, start_time_stim)[0]
    idx_stop_stim = find_nearest(lfp.time.values, stop_time_stim)[0]
    
    return lfp[idx_start_stim:idx_stop_stim]

def subtract_lfp_baseline_all_sims(lfp_orig, tstim_onset = 250, contributions = False, contributions_summed = False):
    '''
    This function subtracts LFP before stimulus onset (baseline) from LFP during stimulus presentations. 
    Parameters
    ---------
            lfp_orig : lfp before subtraction of baseline
            tstim_onset : time at which the stimulus is presented (ms)
            contributions : boolean that indicates whether this is cell type population contributions to total LFP
            contributions_summed : boolean that indicates whether this is population (contributions from excitatory
                                   and inhibitory cell types in each layer already summed) contributions to total LFP
        Returns
        ---------
            lfp_out : local field potential after subtracting baseline
    '''
        
    #TODO: Fix the stupid data organization that forces all these if-statements
    
    lfp_out = dict()

    if contributions:
        for sim_name in lfp_orig.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name].keys():
                lfp_dict = dict()
                
                lfp_temp = lfp_orig[sim_name][pop_name]['trial_avg'].T
                
                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
            
                lfp_dict['trial_avg'] = lfp_temp.T
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name][pop_name]['all_trials'].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name][pop_name]['all_trials'][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
            
    elif contributions_summed:
        for sim_name in lfp_orig.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name]['all_trials'].keys():
                lfp_dict = dict()
                
                lfp_temp = lfp_orig[sim_name]['trial_avg'][pop_name].T
                
                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
            
                lfp_dict['trial_avg'] = lfp_temp.T
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name]['all_trials'][pop_name].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name]['all_trials'][pop_name][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
    else:
        for sim_name in lfp_orig.keys():
            lfp_dict = dict()
            print(sim_name)
            lfp_temp = lfp_orig[sim_name]['trial_avg'].T

            lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)

            lfp_dict['trial_avg'] = lfp_temp.T

            lfp_trials_temp = []
            for itrial in range(lfp_orig[sim_name]['all_trials'].shape[1]):
                lfp_temp = lfp_orig[sim_name]['all_trials'][:,itrial].T

                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)

                lfp_trials_temp.append(lfp_temp.T)
                
            lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
            lfp_out[sim_name] = lfp_dict
            
    return lfp_out

def compute_csd(lfp, method = 'delta', gauss_filter = (1.4,0), coord_electrodes = np.linspace(0,1000E-6,26) * pq.m,\
                diam = 800E-6 * pq.m, sigma = 0.3*pq.S/pq.m, sigma_top = 0.3*pq.S/pq.m, h = 40*1E-6*pq.m, mode = 'sim'):
    
    '''
    This function computes CSD from LFP using the delta iCSD (https://doi.org/10.1016/j.jneumeth.2005.12.005)
    Parameters
    ---------
        lfp : local field potential
        method : method by which CSD is calculated in iCSD
        gauss_filter : smoothing parameter, given in sigma
        coord_electrodes : depth of electrodes on probe
        diam : diameter of laterally constant CSD assumed
        sigma : conductivity in extracellular medium
        sigma_top : conductivity in extracellular medium at top channel
        h : spacing between electrodes
        mode : indicates whether it is calculated for simulation or experimental LFP
    Returns
    ---------
        csd : current source density
    '''

    # simulation LFP is given in mV, while experimental LFP is given in V
    if mode == 'sim':
        lfp = lfp*1E-3*pq.V
    elif mode == 'exp':
        lfp = lfp*pq.V
    else:
        lfp = lfp*1E-3*pq.V
    
    delta_input = {
    'lfp' : lfp,
    'coord_electrode' : coord_electrodes,
    'diam' : diam,
    'sigma' : sigma,
    'sigma_top' : sigma_top,
    'f_type' : 'gaussian',  # gaussian filter. Not used
    'f_order' : (0, 0),     # 3-point filter
    }
    
    step_input = {
    'lfp' : lfp,
    'coord_electrode' : coord_electrodes,
    'diam' : diam,
    'h' : h,
    'sigma' : sigma,
    'sigma_top' : sigma_top,
    'tol' : 1E-12,          # Tolerance in numerical integration
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
    }
    
    spline_input = {
    'lfp' : lfp,
    'coord_electrode' : coord_electrodes,
    'diam' : diam,
    'sigma' : sigma,
    'sigma_top' : sigma_top,
    'num_steps' : len(coord_electrodes)*4,      # Spatial CSD upsampling to N steps
    'tol' : 1E-12,
    'f_type' : 'gaussian',
    'f_order' : (20, 5),
    }

    if method == 'delta':
        csd_dict = dict(
            delta_icsd = icsd.DeltaiCSD(**delta_input)
        )
    elif method == 'step':
        csd_dict = dict(
            step_icsd = icsd.StepiCSD(**step_input)
        )
    elif method == 'spline':
        csd_dict = dict(
            spline_icsd = icsd.SplineiCSD(**spline_input)
        )
        
    #TODO: Set up the input for the other methods
    '''elif method == 'step':
        step_icsd = icsd.StepiCSD(**step_input),
    elif method == 'spline':
        spline_icsd = icsd.SplineiCSD(**spline_input),
    elif method == 'standard':
        std_csd = icsd.StandardCSD(**std_input),'''
  

    for method_, csd_obj in list(csd_dict.items()):
        csd_raw = csd_obj.get_csd()
        
    # Converting from planar to volume density
    if method == 'delta':
        csd_raw = csd_raw / h
        
    # Apply spatial filtering
    csd = gaussian_filter(csd_raw, sigma = gauss_filter)*csd_raw.units
    
    return csd