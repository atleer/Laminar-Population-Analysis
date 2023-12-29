import numpy as np
import os, sys
import quantities as pq
from glob import glob
import h5py
import scipy.signal as ss
from scipy.ndimage import gaussian_filter
sys.path.append('icsd_scripts/')
import icsd

def get_ecp_raw(path, contributions = False):
    
    all_paths = glob(path)
    
    if contributions:
        filename = 'ecp_contributions.h5'
    else:
        filename = 'ecp.h5'
    
    result = dict()
    for path in all_paths:
        for root, dirs, files in os.walk(path):

            if len(glob(os.path.join(root, filename)))>0:

                file_path = glob(os.path.join(root,filename))[0]

                sim_name = root.split('/')[-1]
                f = h5py.File(file_path, 'r')
                primary_keys = list(f.keys())

                print(sim_name)
                #print('primary_keys: ', primary_keys)
                if contributions:
                    ecp_contribution_raw = dict()
                    for cell_pop_name in primary_keys:
                        ecp_contribution_raw[cell_pop_name] = np.array(f[cell_pop_name])

                    result[sim_name] = ecp_contribution_raw
                else:
                    #secondary_keys = list(f[primary_keys[0]].keys())

                    #print('secondary_keys: ', secondary_keys)
                    
                    if primary_keys[0] == 'ecp':
                        ecp_raw = np.array(f['ecp']['data'])
                    else:
                        ecp_raw = np.array(f['data'])

                    result[sim_name] = ecp_raw

                f.close()
            
    return result

def get_lfp_all_sims(files,trial_dur=1500,filter_order=2,fs=10000,fc=500,q=10,downsample=True,contributions=False,do_filter=True, t_rem_trial_start = 0):
    '''
    Arguments:
            files: dictionary containing ecp from different simulations
    Output:
            result: dictionary containing lfp from different simulations
    '''
    result = dict()
    
    for sim_name in files.keys():
        print(sim_name)
        
        ecp_raw = files[sim_name]
        
        if contributions:
            lfp_pop_dict = dict()
            for cell_pop_name in ecp_raw.keys():
                lfp_dict = dict()
                print(cell_pop_name)
                lfp = get_lfp(ecp_raw[cell_pop_name], q = 10, downsample = downsample, do_filter = do_filter)
                                
                n_trials = lfp.shape[0] // trial_dur
                
                lfp_reshaped = np.zeros((lfp.shape[1],n_trials,trial_dur))
                
                print(n_trials)
                
                for itrial in range(n_trials):
                    lfp_reshaped[:, itrial] = lfp[itrial*trial_dur:(itrial+1)*trial_dur,:].T
                        
                # TODO: Figure out which reshape order to use
                #lfp_reshaped = lfp.reshape(trial_dur, n_trials, nchan).T
                
                lfp_dict['all_trials'] = lfp_reshaped[:,:,t_rem_trial_start:]

                lfp_trial_avg = np.mean(lfp_reshaped[:,:,t_rem_trial_start:], axis = 1)

                lfp_dict['trial_avg'] = lfp_trial_avg
                
                lfp_pop_dict[cell_pop_name] = lfp_dict

            result[sim_name] = lfp_pop_dict
        else:
            lfp_dict = dict()
                        
            lfp = get_lfp(ecp_raw, filter_order = filter_order, fs = fs, fc = fc, q = q, downsample = downsample, do_filter = do_filter)

            n_trials = lfp.shape[0] // trial_dur
            #n_trials = 7
                                    
            lfp_reshaped = np.zeros((n_trials,lfp.shape[1],trial_dur))
                
            for itrial in range(n_trials):
                lfp_reshaped[itrial] = lfp[itrial*trial_dur:(itrial+1)*trial_dur,:].T
    
            # TODO: Figure out which reshape order to use
            #lfp_reshaped = lfp.reshape(trial_dur, n_trials, nchan).T
            
            lfp_dict['all_trials'] = lfp_reshaped[:,:,t_rem_trial_start:]

            lfp_trial_avg = np.mean(lfp_reshaped[:,:,t_rem_trial_start:], axis = 0)

            lfp_dict['trial_avg'] = lfp_trial_avg

            result[sim_name] = lfp_dict

    return result

def get_mua(ecp, filter_order = 5, fs = 10000, fc = 300, q = 20, downsample = True):
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
                lfp : local field potential
    '''
    # creating high-pass filter
    Wn = fc/fs/2

    b, a = ss.butter(filter_order, Wn, btype = 'high')

    mua = ss.filtfilt(b, a, ecp, axis = 0)

    if downsample:
        for q_ in [10, q // 10]:
            mua = ss.decimate(mua, q_, axis = 0)
    
    return abs(mua)

def find_files(path, filename = 'lfp.npy'):
    
    all_paths = glob(path)
    
    result = dict()
    for path in all_paths:
        for root, dirs, files in os.walk(path):
            if len(glob(os.path.join(root, filename)))>0:

                file_path = glob(os.path.join(root,filename))[0]
                sim_name = file_path.split('/')[-2]
                try:
                    file = np.load(file_path, allow_pickle=True)[()]
                except:
                    print('File could not be loaded')
                result[sim_name] = file
            
    return result

def find_all_fir_rates_files_sim(path,filename):
    
    result = dict()
    for root, dirs, files in os.walk(path):
        if len(glob(os.path.join(root,filename)))>0:
            
            fir_rate_file_path = glob(os.path.join(root,filename))[0]
            sim_name = fir_rate_file_path.split('/')[-2]
            fir_rate_file = np.load(fir_rate_file_path, allow_pickle=True)[()]
            result[sim_name] = fir_rate_file
    return result

def get_ecp_raw(path, contributions = False):
    
    all_paths = glob(path)
    
    if contributions:
        filename = 'ecp_contributions.h5'
    else:
        filename = 'ecp.h5'
    
    result = dict()
    for path in all_paths:
        for root, dirs, files in os.walk(path):

            if len(glob(os.path.join(root, filename)))>0:

                file_path = glob(os.path.join(root,filename))[0]

                sim_name = root.split('/')[-1]
                f = h5py.File(file_path, 'r')
                primary_keys = list(f.keys())

                print(sim_name)
                #print('primary_keys: ', primary_keys)
                if contributions:
                    ecp_contribution_raw = dict()
                    for cell_pop_name in primary_keys:
                        ecp_contribution_raw[cell_pop_name] = np.array(f[cell_pop_name])

                    result[sim_name] = ecp_contribution_raw
                else:
                    #secondary_keys = list(f[primary_keys[0]].keys())

                    #print('secondary_keys: ', secondary_keys)
                    
                    if primary_keys[0] == 'ecp':
                        ecp_raw = np.array(f['ecp']['data'])
                    else:
                        ecp_raw = np.array(f['data'])

                    result[sim_name] = ecp_raw

                f.close()
            
    return result

def get_lfp(ecp, filter_order = 5, fs = 10000, fc = 300, q = 20, downsample = True, do_filter = True):
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

def timestamps_to_spiketrains_sim(timestamps_configs, mode = 'exc_and_inh'):

    sims_spike_trains_cell_pops = dict()
    for sim_name in timestamps_configs.keys():
        print('\n',sim_name)

        timestamps_all_cell_pops = timestamps_configs[sim_name]
        cell_names = list(timestamps_all_cell_pops.keys())


        sim_spike_trains_cell_pops = dict()

        for icell, cell_name in enumerate(cell_names):
            print(cell_name)
            if mode == 'only_exc':
                if cell_name[0] != 'e':
                    print('Use only excitatory cells. Skip.')
                    continue
            elif mode == 'only_exc_except_L1':
                if cell_name[0] != 'e' and cell_name[1] != '1':
                    print('Use only excitatory cells except for L1. Skip.')
                    continue
            timestamps_cell_pop = timestamps_all_cell_pops[cell_name]

            neuron_ids = timestamps_cell_pop['neuron_ids_of_timestamps']
            trials = timestamps_cell_pop['trials']
            timestamps = timestamps_cell_pop['timestamps']
            #timestamps = timestamps_cell_pop['timestamps_by_trial']
            trial_dur_sim = timestamps_cell_pop['trial_dur']
            if icell == 0:
                ntrials_sim = len(np.unique(trials))
            '''TODO: Implement general way to determine # of trials (that also works for sims that lasted a bit too long)'''
            #ntrials_sim = 10
            timestamps = timestamps % trial_dur_sim
            all_neuron_ids = timestamps_cell_pop['all_neuron_ids_in_pop']

            spike_trains = np.zeros((len(all_neuron_ids), ntrials_sim, trial_dur_sim))
            bins = np.arange(0,trial_dur_sim+1,1)

            for i_nrn, neuron_id in enumerate(all_neuron_ids):
                mask_neuron_id = neuron_ids == neuron_id

                timestamps_nrn = timestamps[mask_neuron_id]

                trials_this_nrn = trials[mask_neuron_id]

                for trial in np.unique(trials_this_nrn):
                    if trial >= ntrials_sim:
                        continue
                    mask_trial = trials_this_nrn == trial
                    timestamps_nrn_this_trial = timestamps_nrn[mask_trial]
                    spike_train_trial, _ = np.histogram(timestamps_nrn_this_trial, bins)

                    spike_trains[i_nrn, int(trial)] = spike_train_trial

            sim_spike_trains_cell_pops[cell_name] = spike_trains[:,:ntrials_sim]

        sims_spike_trains_cell_pops[sim_name] = sim_spike_trains_cell_pops
        
    return sims_spike_trains_cell_pops
    
def spiketrains_by_layer(sims_spike_trains_cell_pops, npop_guess = 4, nrn_type = 'exc_and_inh'):

    sims_spike_trains_layer_pops = dict()

    for sim_name in sims_spike_trains_cell_pops.keys():
        print(sim_name)

        sim_spike_trains_layer_pops = dict()
        
        if npop_guess == 5:
            if nrn_type == 'exc_and_inh':
                pop_key_L1 = 'L1'
                pop_key_L23 ='L2/3'
                pop_key_L4 = 'L4'
                pop_key_L5 = 'L5'
                pop_key_L6 = 'L6'
            elif nrn_type == 'exc':
                pop_key_L23 ='E2/3'
                pop_key_L4 = 'E4'
                pop_key_L5 = 'E5'
                pop_key_L6 = 'E6'
            elif nrn_type == 'inh':
                pop_key_L1 = 'I1'
                pop_key_L23 ='I2/3'
                pop_key_L4 = 'I4'
                pop_key_L5 = 'I5'
                pop_key_L6 = 'I6'
            sim_spike_trains_layer_pops[pop_key_L1] = []
            sim_spike_trains_layer_pops[pop_key_L23] = []
            sim_spike_trains_layer_pops[pop_key_L4] = []
            sim_spike_trains_layer_pops[pop_key_L5] = []
            sim_spike_trains_layer_pops[pop_key_L6] = []
        if npop_guess == 4:
            if nrn_type == 'exc_and_inh':
                pop_key_L23 = 'L2/3'
                pop_key_L4 = 'L4'
                pop_key_L5 = 'L5'
                pop_key_L6 = 'L6'
            elif nrn_type == 'exc':
                pop_key_L23 = 'E2/3'
                pop_key_L4 = 'E4'
                pop_key_L5 = 'E5'
                pop_key_L6 = 'E6'
            elif nrn_type == 'inh':
                pop_key_L23 = 'I2/3'
                pop_key_L4 = 'I4'
                pop_key_L5 = 'I5'
                pop_key_L6 = 'I6'
            sim_spike_trains_layer_pops[pop_key_L23] = []
            sim_spike_trains_layer_pops[pop_key_L4] = []
            sim_spike_trains_layer_pops[pop_key_L5] = []
            sim_spike_trains_layer_pops[pop_key_L6] = []
        elif npop_guess == 3:
            pop_key_L23 = 'L2/3&L4'
            pop_key_L4 = 'L2/3&L4'
            sim_spike_trains_layer_pops['L2/3&L4'] = []
            sim_spike_trains_layer_pops[pop_key_L5] = []
            sim_spike_trains_layer_pops[pop_key_L6] = []
        elif npop_guess == 2:
            pop_key_L23 = 'upper_layers'
            pop_key_L4 = 'upper_layers'
            pop_key_L5 = 'deep_layers'
            pop_key_L6 = 'deep_layers'
            sim_spike_trains_layer_pops['upper_layers'] = []
            sim_spike_trains_layer_pops['deep_layers'] = []
        elif npop_guess == 1:
            pop_key_L23 = 'v1'
            pop_key_L4 = 'v1'
            pop_key_L5 = 'v1'
            pop_key_L6 = 'v1'
            sim_spike_trains_layer_pops['v1'] = []
            

        for cell_pop_name, sim_spike_trains_cell_pop in sims_spike_trains_cell_pops[sim_name].items():
            print(cell_pop_name, sims_spike_trains_cell_pops[sim_name][cell_pop_name].shape[0])
            if npop_guess > 4:
                if nrn_type == 'exc_and_inh':
                    condition = cell_pop_name[1] == '1'
                elif nrn_type == 'exc':
                    condition = np.logical_and(cell_pop_name[1] == '1', cell_pop_name[0] == 'e')
                elif nrn_type == 'inh':
                    condition = np.logical_and(cell_pop_name[1] == '1', cell_pop_name[0] == 'i')
                if condition:
                    if len(sim_spike_trains_layer_pops[pop_key_L1]) == 0:
                        sim_spike_trains_layer_pops[pop_key_L1] = sim_spike_trains_cell_pop
                    else:
                        sim_spike_trains_layer_pops[pop_key_L1] = np.concatenate((sim_spike_trains_layer_pops[pop_key_L1], 
                                                                       sim_spike_trains_cell_pop), axis = 0)

            if nrn_type == 'exc_and_inh':
                condition = cell_pop_name[1] == '2'
            elif nrn_type == 'exc':
                condition = np.logical_and(cell_pop_name[1] == '2', cell_pop_name[0] == 'e')
            elif nrn_type == 'inh':
                condition = np.logical_and(cell_pop_name[1] == '2', cell_pop_name[0] == 'i')
            if condition:
                if len(sim_spike_trains_layer_pops[pop_key_L23]) == 0:
                    sim_spike_trains_layer_pops[pop_key_L23] = sim_spike_trains_cell_pop
                else:
                    sim_spike_trains_layer_pops[pop_key_L23] = np.concatenate((sim_spike_trains_layer_pops[pop_key_L23], 
                                                                   sim_spike_trains_cell_pop), axis = 0)

            if nrn_type == 'exc_and_inh':
                condition = cell_pop_name[1] == '4'
            elif nrn_type == 'exc':
                condition = np.logical_and(cell_pop_name[1] == '4', cell_pop_name[0] == 'e')
            elif nrn_type == 'inh':
                condition = np.logical_and(cell_pop_name[1] == '4', cell_pop_name[0] == 'i')
            if condition:
                if len(sim_spike_trains_layer_pops[pop_key_L4]) == 0:
                    sim_spike_trains_layer_pops[pop_key_L4] = sim_spike_trains_cell_pop
                else:
                    sim_spike_trains_layer_pops[pop_key_L4] = np.concatenate((sim_spike_trains_layer_pops[pop_key_L4], 
                                                                   sim_spike_trains_cell_pop), axis = 0)

            if nrn_type == 'exc_and_inh':
                condition = cell_pop_name[1] == '5'
            elif nrn_type == 'exc':
                condition = np.logical_and(cell_pop_name[1] == '5', cell_pop_name[0] == 'e')
            elif nrn_type == 'inh':
                condition = np.logical_and(cell_pop_name[1] == '5', cell_pop_name[0] == 'i')
            if condition:
                if len(sim_spike_trains_layer_pops[pop_key_L5]) == 0:
                    sim_spike_trains_layer_pops[pop_key_L5] = sim_spike_trains_cell_pop
                else:
                    sim_spike_trains_layer_pops[pop_key_L5] = np.concatenate((sim_spike_trains_layer_pops[pop_key_L5], 
                                                                   sim_spike_trains_cell_pop), axis = 0)

            if nrn_type == 'exc_and_inh':
                condition = cell_pop_name[1] == '6'
            elif nrn_type == 'exc':
                condition = np.logical_and(cell_pop_name[1] == '6', cell_pop_name[0] == 'e')
            elif nrn_type == 'inh':
                condition = np.logical_and(cell_pop_name[1] == '6', cell_pop_name[0] == 'i')
            if condition:
                if len(sim_spike_trains_layer_pops[pop_key_L6]) == 0:
                    sim_spike_trains_layer_pops[pop_key_L6] = sim_spike_trains_cell_pop
                else:
                    sim_spike_trains_layer_pops[pop_key_L6] = np.concatenate((sim_spike_trains_layer_pops[pop_key_L6], 
                                                                   sim_spike_trains_cell_pop), axis = 0)

        sims_spike_trains_layer_pops[sim_name] = sim_spike_trains_layer_pops
    
    return sims_spike_trains_layer_pops


def subtract_lfp_baseline_all_sims(lfp_orig, tstim_onset = 250, contributions = False, contributions_summed = False):
        
    #TODO: Fix the stupid data organization that forces all these if-statements
    
    lfp_out = dict()

    if contributions:
        for sim_name in lfp.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name].keys():
                lfp_dict = dict()
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name][pop_name]['all_trials'].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name][pop_name]['all_trials'][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[(tstim_onset-int(tstim_onset/2)):tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict['trial_avg'] = np.mean(lfp_dict['all_trials'], axis = 0)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
            
    elif contributions_summed:
        for sim_name in lfp.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name][()]['all_trials'].keys():
                lfp_dict = dict()
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name][()]['all_trials'][pop_name].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name][()]['all_trials'][pop_name][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict['trial_avg'] = np.mean(lfp_dict['all_trials'], axis = 0)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
    else:
        for sim_name in lfp_orig.keys():
            lfp_dict = dict()
            print(sim_name)

            lfp_trials_temp = []
            for itrial in range(lfp_orig[sim_name]['all_trials'].shape[1]):
                lfp_temp = lfp_orig[sim_name]['all_trials'][:,itrial].T
                #for ichan in range(lfp_temp.shape[0]):
                #    lfp_temp -= np.mean(lfp_temp[ichan, :tstim_onset])
                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)

                lfp_trials_temp.append(lfp_temp.T)
                
            lfp_dict['all_trials'] = np.array(lfp_trials_temp)
            
            lfp_dict['trial_avg'] = np.mean(lfp_dict['all_trials'], axis = 0)
                
            lfp_out[sim_name] = lfp_dict
            
    return lfp_out

def compute_csd(lfp, method = 'delta', gauss_filter = (1.4,0), coord_electrodes = np.linspace(0,1000E-6,26) * pq.m,\
                diam = 800E-6 * pq.m, sigma = 0.3*pq.S/pq.m, sigma_top = 0.3*pq.S/pq.m, h = 40*1E-6*pq.m, mode = 'sim',\
               vaknin_el = True):
    
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
    'f_type' : 'gaussian',  # gaussian filter
    'f_order' : (0, 0),     # 3-point filter, sigma = 1.
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
    
    std_input = {
    'lfp' : lfp,
    'coord_electrode' : coord_electrodes,
    'sigma' : sigma,
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
    'vaknin_el' : vaknin_el,
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
    elif method == 'standard':
        csd_dict = dict(
            std_icsd = icsd.StandardCSD(**std_input)
        )
        
    #TODO: Set up the input for the other methods
    '''elif method == 'step':
        step_icsd = icsd.StepiCSD(**step_input),
    elif method == 'spline':
        spline_icsd = icsd.SplineiCSD(**spline_input),
    elif method == 'standard':
        std_csd = icsd.StandardCSD(**std_input),'''
  

    for method_, csd_obj in list(csd_dict.items()):
        #csd_raw = (csd_obj.filter_csd(csd_obj.get_csd()))
        csd_raw = csd_obj.get_csd()
        
    # Converting from planar to volume density
    #print(method, csd_raw)
    if method == 'delta' or method == 'standard':
        csd_raw = csd_raw / h
        #print(csd_raw)
        
    # Apply spatial filtering
    csd_smooth = gaussian_filter(csd_raw, sigma = gauss_filter)*csd_raw.units
    #csd_smooth = csd_raw
    
    return csd_smooth