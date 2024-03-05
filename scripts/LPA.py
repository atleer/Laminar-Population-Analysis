'''
This file should contain most tools used for LPA.
'''

import pylab as pl
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint, Bounds

try:
    import openopt as oopt
    oopt_import = True
except ImportError:
    oopt_import = False

oopt_import_msg = "WARNING: No module named 'openopt'. LPA must " \
    "be done without using 'fmin_oopt()' method and associated "\
    "solvers. This means that only 'randw' algorithm is available."


glp_solvers = ['galileo' ,'pswarm', 'de', 'stogo', 'isres', 'mlsl']
nlp_solvers = ['ralg', 'scipy_cobyla', 'algencan', 'scipy_slsqp',
               'mma', 'auglag', 'ipopt', 'lincher', 'scipy_fmin',
               'scipy_lbfgsb']


class LPA_Signal(object):
    '''
    Main class for Laminar Population Analysis
    '''
    def __init__( self, mua_data, dt, lfp_data=None,
                  z_start=0., z_space=0.1, casename='', rneg_factor=1E6,
                  tstim=0, sub_at='base',
                  verbose=False, pen_fac=0):
        '''
        This function ...

        '''
        
        if verbose:
            msg = 'This is class *LPA_Signal* in *pyLPA* module'
            print(msg)
        
        if not oopt_import:
            print(oopt_import_msg)
            
        self.nstim, self.ntime, self.nchan = pl.asarray(mua_data).shape

        self.dt = dt
        self.z_start = z_start
        self.z_space = z_space
        self.el_coords = pl.arange(z_start, z_start+self.nchan*z_space, z_space)
        #print('self.el_coords', self.el_coords)
        self.rneg_factor = rneg_factor
        self.tstim = tstim
        self.sub_at = sub_at
        self.verbose = verbose
        self.err_wo_pen = 0
        self.tau = 0
        self.delta = 0

        self.tstim_idx = pl.floor(self.tstim/self.dt)
        
        # create matrices and calculate variances
        self.importDataset(mua_data, 'MUA')
        if not np.any(lfp_data==None):
            self.importDataset(lfp_data, 'LFP')
            self.mode = 'lfp'
        else:
            self.mode = 'mua'

    def importDataset( self, lpadata, mode ):
        '''
        This function ...
                reshapes matrices and computes baseline variance and adds to attributes
    
        Arguments
                lpadata: signal to be decomposed, MUA or LFP
                mode:    string saying whether the signal is MUA or LFP
        
        '''

        matname = mode.lower() + 'mat'
        varname = mode.lower() + 'var'

        nstim, ntime, nchan = lpadata.shape

        if mode.lower()=='lfp' and hasattr(self, '_muamat'):
            if nstim != self.nstim:
                raise(Exception, 'Number of stimuli in %s and MUA data' \
                    ' does not match' % setname)
            elif ntime != self.ntime:
                raise(Exception, 'Number of sampling points in %s and' \
                    ' MUA data does not match' % mode)
            elif nchan != self.nchan:
                raise(Exception, 'Number of channels in %s and MUA data' \
                    ' does not match' % mode)
        
        # Apply base/mean subtraction to each stimulus and channel separately
        tmp_idx = 0
        if self.sub_at=='base':
            tmp_idx = self.tstim_idx
        elif self.sub_at=='mean':
            tmp_idx = ntime
        else:
            msg = '%s is not a valid choice for sub_at' % self.sub_at
        
        # subtract baseline
        if not tmp_idx==0:
            for istim in range(nstim):
                for ichan in range(nchan):
                    lpadata[istim, :, ichan] = lpadata[istim, :, ichan] - np.mean(lpadata[istim, :int(tmp_idx), ichan])

        # reshape data to 2D
        lpamat = self._reshapeMat( lpadata )
        #lpamat = lpadata.reshape((self.nstim*self.ntime, self.nchan)).transpose()
        
        # Evaluate variances of stimulus evoked signal
        lpavar = lpadata[:, int(self.tstim_idx):, :].var()

        exec('self._'+matname+' = lpamat')
        exec('self._'+varname+' = lpavar')

    def __call__( self, mode, solver, x0, lb, ub, init_args={}, solve_args={},
                  f_args=(), plot=False, pen_fac=0):
        '''
        This is where the action starts.
    
        Arguments
        --------

        Keyword arguments
        -----------------
        '''
        
        if self.verbose:
            msg = 'Solving for %s part of signal' % mode 
            print(msg)
        # Check if initial guess is provided in one or both of the argument
        # dictionaries
        redundant_args = ['A', 'b', 'lb', 'ub', 'x0', 'plot']
        for red_arg in redundant_args:
            if red_arg in init_args.keys():
                msg = 'Initial guess *%s* found in *init_args*. I will ' \
                    'override this value with the value *%s* passed on as '\
                    'positional/keyword argument' % (red_arg, red_arg)
                if self.verbose:
                    print(msg)
                del init_args[red_arg]
            if red_arg in solve_args.keys():
                msg = 'Initial guess *%s* found in *solve_args*. I will ' \
                    'override this value with the value *%s* passed on as '\
                    'positional/keyword argument' % (red_arg, red_arg)
                if self.verbose:
                    print(msg)
                del solve_args[red_arg]        

        # Set temporary attributes. These are removed in the end
        attr_list = ['solver', 'x0', 'lb', 'ub', 'init_args', 'solve_args']
        for attr_name in attr_list:
            try:
                exec('self.'+attr_name+' = '+attr_name+'.copy()')
            except AttributeError:
                exec('self.'+attr_name+' = '+attr_name)

        if mode.lower()=='mua':
            # Check for consistent length of x0
            M_params = pl.asarray(x0).squeeze()
            if pl.mod(M_params.shape[0], 3):
                err_msg = 'Number of elements in x0 must be divisable' \
                    'by 3. Current number of elements is %d' % M_params.shape[0]
                raise(Exception, err_msg)
            else:
                self.npop = int(M_params.shape[0]/3)
                npop = self.npop
            
            # Do some pre-solve operations (nothing really happens here)
            f_args = self._muaWarmup(*f_args)
                
            # create linear constraints
            A, b = self._muaConstraints()         
            
            # Set the error function
            f_fnc = self._muaErr
            '''
            print('A: ', A, '\n', 'b: ', b, '\n', 'lb: ', lb, '\n', 'ub: ', ub)
            print('ub[:npop]: ', ub[:npop])
            print('ub[npop:2*npop]: ', ub[npop:2*npop])
            print('ub[2*npop:]: ', ub[2*npop:])
            '''
        elif mode.lower()=='lfp':
            # linear constraints
            f_args = self._lfpWarmup(*f_args)
            A, b = self._lfpConstraints()
            print('len(ub): ', len(ub))
            self.npop = len(ub) // 2
            npop = self.npop
            # Error function
            f_fnc = self._lfpErr
        self.pen_fac = pen_fac
        print('self.pen_fac: ', self.pen_fac)

        init_args_internal = {
            'A' : A, 'b' : b,
            'lb' : self.lb, 'ub' : self.ub
            }
        
        # Override self.init_args with init_args_internal
        self.init_args.update(init_args_internal)
        
        #print('self.init_args: ', self.init_args)
        #print('init_args: ', init_args)

        # Finally, put everything into a dictionary that can be 
        # passed to self._solve()
        #print(self.solve_args)
        
        fmin_args = {
            'solver' : self.solver,
            'f_fnc' : f_fnc,
            'f_args' : f_args,
            'x0' : self.x0,
            'init_args' : self.init_args,
            'solve_args' : self.solve_args,
            'plot' : plot
            }
        
        
        #res = self._solve(fmin_args)

        if mode == 'mua':
            #if you want to use openopt as solver:
            res = self._solve(fmin_args)
            '''
            
            Constraints if you want to use scipy.optimize.minimize as solver (performs poorly):
            cons = [{'type': 'ineq', 'fun': lambda x:  -1*x[:npop] + ub[:npop]},
                    {'type': 'ineq', 'fun': lambda x:  -1*x[npop:2*npop] + ub[npop:2*npop]},
                    {'type': 'ineq', 'fun': lambda x:  -1*x[2*npop:] + ub[2*npop:]}]
            
            using scipy.optimize.minimize as solver
            #res = minimize(f_fnc, self.x0, args = f_args, constraints = cons, method = 'COBYLA')
            
            ### Using scipy.optimize.di
            erential_evolution as solver:
            ### please note: it can sometimes switch up the indices of the populations even if it finds the correct minimum
            bounds = [[]]*len(ub)
            maxpos = np.linspace(1/self.npop, 1, self.npop)
            ub[:self.npop] = ub[:self.npop]*maxpos
            print("ub = ", ub)
            for i in range(len(ub)):
                bounds[i] = (lb[i], ub[i])
                
            res = differential_evolution(f_fnc, bounds, args = f_args, maxiter = 150)'''
        else:
            # lfp. use scipy.optimize.minimize as solver
            #cons = [{'type': 'ineq', 'fun': lambda x:  -1*x[0] + ub[0]},
            #        {'type': 'ineq', 'fun': lambda x:  -1*x[1] + ub[1]}]
            
            #bounds = [[lb[0],ub[0]], [lb[1],ub[1]], [lb[2],ub[2]] , [lb[3],ub[3]],[lb[4],ub[4]],[lb[5],ub[5]]]
            #bounds = Bounds([lb, lb, lb], [ub, ub, ub])
            #res = differential_evolution(f_fnc, bounds, args = f_args, polish = False)
            #res = differential_evolution(f_fnc, bounds = bounds)
            #res = minimize(f_fnc, self.x0, bounds = bounds, args = f_args, method='Nelder-Mead', jac = None, tol = 1E-6, options={'maxiter': 1E6, 'disp': True})#, constraints = cons)
            res = self._solve(fmin_args)
            
        # res.xf is the optimized parameters if you use openopt and res.x is the optimized parameters if you use scipy

        ############################################################
        # Post solver processing
        # Find decompositions and full matrix for best parameters
        
        # h_list: values of temporal kernel
        h_list = 0
        if mode.lower()=='mua':
            # Smat is the spatial profile, Mmat, and Tmat_tmp is the firing rate, rmat
            Smat, Tmat_tmp = self.muaDecomp( res.xf, *f_args )
        elif mode.lower()=='lfp':
            # Smat is the spatial profile, Lmat, and Tmat_tmp is the firing rate convolved with the time kernel
            Smat, Tmat_tmp, h_list = self.lfpDecomp( res.xf, *f_args )
            #Smat, Tmat_tmp, h_list = self.lfpDecomp( res.x, *f_args )
        
        
        # calculating the predicted MUA or LFP - Phi
        Phimat_tmp = pl.dot(Smat, Tmat_tmp)
        # Reshape to 3d
        Tmat = self._reshapeMat( Tmat_tmp )
        Phimat = self._reshapeMat( Phimat_tmp )
        
        # deleting temporary attributes
        for attr_name in attr_list:
            exec('del self.'+attr_name)        
        
        return res, Smat, Tmat, Phimat, h_list, self.err_wo_pen, self.tau, self.delta

    def _reshapeMat( self, rawmat ):
        '''
        This function ...
                reshapes the data between 2D and 3D
        
        Arguments:
                Rawmat: the LFP or MUA data

        Output:
                outmat: the reshaped data matrix
        '''
        
        if rawmat.ndim==2:
            
            # Create the 3D scores
            outmat = rawmat.transpose().reshape( 
                (self.nstim, self.ntime, rawmat.shape[0]) )
            # Create full 3D array
#            Phimat = pl.asarray(
#                [pl.dot(Smat, Tm.transpose()).transpose() for Tm in Tmat])
        elif rawmat.ndim==3:
            outmat = rawmat.reshape(
                (self.nstim*self.ntime, rawmat.shape[-1]) ).transpose()
            

        return outmat
        

    def _solve( self, fmin_args ):
        '''
        This function ...
                just tells you that openopt isn't installed if you haven't, otherwise it calls the function which implements the                 openopt solver
    
        Arguments
                fmin_args: dictionary of arguments

        Output:
                r:         optimized parameters and more info from the solver
        '''
        
        if fmin_args['solver'] == 'fmin_randw':
            r = self._fminRandw()
        elif not oopt_import:
            print(oopt_import_msg) #raise(Exception, oopt_import_msg)
        elif fmin_args['solver'] in nlp_solvers or glp_solvers:
            r = self._fminOopt(**fmin_args)
        else:
            pass
        

        return r
    
    def _fminRandw( self ):
        '''
        This function ...
            is not used
    
        Aguments
        --------

        Keyword arguments
        -----------------
        '''

        pass

    def _fminOopt( self, f_fnc, f_args, solver, x0, init_args={},
                    solve_args={}, plot=False, pen_fac=0):
        '''
        This function ...
                sets up and implements the openopt solver
        Arguments:
                f_fnc:      the function to be minimized, which is the error evaluation here
                f_args:     arguments to the function that comes in addition to the parameters to be optimized
                init_args:  initial values and constraints
                solve_args: specifiesd whether to plot or not

        Output:
                r:          optimized parameters and more info from the solver
        '''
        
        fnc = lambda x: f_fnc(x, *f_args)

        # Check what solver is in use and put initial guess argument in right
        # dictionary
        if solver in glp_solvers:
            solve_args['plot'] = plot
            solve_args['x0'] = x0
            
            p = oopt.GLP(fnc, **init_args) # Set up openopt class
            r = p.solve(solver, **solve_args) # and solve

        elif solver in nlp_solvers:
            init_args['plot'] = plot
            init_args['x0'] = x0

            p = oopt.NLP(fnc, **init_args) # Set up openopt class
            r = p.solve(solver, **solve_args) # and solve
            
        else: 
            raise(Exception, 'The solver %s is not recognized. Check' \
                'spelling!' % solver)

        
        return r

    def _muaErr( self, M_params ):
        '''
        This function ...
                calls the mua decomposition function to get the spatial profile, M, and the firing rates and then passes these                 to the error evaluation function
        Arguments
                M_params: the parameters to be optimized; the position and shape of the spatial profiles

        Output:
                ff:       relative mean square error between the MUA prediction and the actual MUA
        '''
        
        Mmat, rmat = self.muaDecomp( M_params )
        ff = self.errEval(self._muamat, Mmat, rmat)

        return ff
        
    def _lfpErr( self, L_params, rmat, kernel ):
        '''
        This function ...
                calls the lfp decomposition function to get the spatial profile, L, and the postsynaptic temporal profile of                   the LFP (firing rate convolved with temporal kernel) and then passes these to the error evaluation function
    
        Aguments
                L_params: parameters of the temporal kernel to be optimized, which is the time constant, tau, and the delay,                             delta
                rmat:     firing rates
                kernel:   type of kernel chosen

        Output:
                ff:       relative mean square error between the LFP prediction and the actual LFP
        '''
        
        Lmat, Rmat, h_list = self.lfpDecomp( L_params, rmat, kernel )
        
        Lmat_norm = Lmat / np.max(Lmat, axis = 0)
        
        #pen_fac_dev_from_0 = 1E2
        #print('pen_fac: ', self.pen_fac)
        pen_csd_dev_from_0 = self.pen_fac*np.abs(Lmat_norm.sum(axis=0)).sum()
        
        #print('Penalty term dev. from 0: ', np.round(pen_csd_dev_from_0, 3))
        
        '''inds_peak_sink = np.argmin(Lmat, axis = 0)
        inds_peak_source = np.argmax(Lmat, axis = 0)
        dists_peaks = np.abs(inds_peak_sink - inds_peak_source)
        
        nchans_allowed_between_peaks = 2 # corresponding to 200 um for 40 um spacing
        pen_fac_dist_peaks = 1E-3
        pen_dist_peaks = 0
        for ipop, dist_peaks in enumerate(dists_peaks):
            if dist_peaks > nchans_allowed_between_peaks:
                pen_dist_peaks += pen_fac_dist_peaks*dist_peaks
                #print('Penalty distance peaks: ', ipop, dist_peaks, np.round(pen_dist_peaks, 2))
                
        pen_dev_0_bot_chans_fb = 0
        pen_fac_dev_0_bot_chans = 1E-5
        pen_dev_0_bot_chans_fb = pen_fac_dev_0_bot_chans*np.sum(np.abs(Lmat[:int(Lmat.shape[0]),-1]), axis = 0)
        #print('Penalty term dev. from 0 bottom channels: ', np.round(pen_dev_0_bot_chans_fb, 3))'''
        
        ff = self.errEval(self._lfpmat, Lmat, Rmat) + pen_csd_dev_from_0# + pen_dist_peaks# + pen_dev_0_bot_chans_fb#
        
        self.err_wo_pen = ff - pen_csd_dev_from_0
        
        #print('ff: ', ff)

        return ff
    
    def lfpDecomp( self, L_params, rmat, kernel ):
        '''
        This function ...
                decomposes the LFP signal into the spatial profile, L, and the postsynaptic temporal profile of the LFP (firing                 rate convolved with temporal kernel)
    
        Arguments:
                L_params: parameters to be optimized; the temporal kernel parameters tau - the time constant - and                               delta - the delay
                rmat:     firing rate
                kernel:   type of kernel chosen
                
        Output:
                Lmat:   spatial profile
                Rmat:   postsynaptic temporal profile of the LFP
                h_list: temporal kernel values 
        '''
        
        if 'unique' in kernel:
            param_string = '(self, L_params, self.dt, self.npop)'
        else:
            param_string = '(L_params, self.dt)'
        #print('kernel: ', kernel, 'L_params: ', L_params, 'dt: ', self.dt)
        #print('self.ub', self.ub, 'self.lb:', self.lb)
        
        #if np.any(L_params[paramNr] < self.lb[paramNr]) or np.any(L_params[paramNr] > self.ub[paramNr]):
        #    for paramNr in np.where(
        
        for paramNr in range(len(self.lb)):
            while L_params[paramNr] < self.lb[paramNr] or  L_params[paramNr] > self.ub[paramNr]:
                #print('Here')
                L_params[paramNr] = np.random.uniform(self.lb[1],self.ub[1])
            
        
        
        
        '''if L_params[1] < self.lb[1] or np.abs(L_params[1]) > self.ub[1]:
            print('here')
            L_params[1] = np.random.uniform(self.lb[1],self.ub[1])
        
        if L_params[0] < self.lb[0] or np.abs(L_params[0]) > self.ub[0]:
            print('There')
            L_params[0] = np.random.uniform(self.lb[0], self.ub[0])'''
        #print('kernel: ', kernel, 'L_params: ', L_params, 'dt: ', self.dt)'''
        #print('L_params:', L_params, ', lb: ', self.lb, ', ub:', self.ub)
        h_list = eval('_'+kernel+param_string)        
        Rmat = _createRmat(h_list, rmat, self.nstim, self.ntime)
        Lmat = pl.dot(self._lfpmat, pl.linalg.pinv(Rmat))

        return Lmat, Rmat, h_list

    def muaDecomp( self, M_params):
        '''
        This function ...
                decomposes the MUA signal into the spatial profile, M, and firing rates of the populations
    
        Arguments:
                M_params: parameters to be optimized, which is the position and shape of the spatial profiles, M
        Output:
                Mmat:   spatial profile
                rmat:   firing rates
        '''

        Mmat = _createMmat( M_params, self.el_coords )
        rmat = _create_rmat( Mmat, self._muamat, self._muavar,
                             self.rneg_factor )
        
        return Mmat, rmat
    
    def _muaConstraints( self ):
        '''
        This function ...
                    sets constraints on the MUA spatial profile
        Arguments
        --------
        
        Keyword arguments
        -----------------
        '''
        
        npop = int(self.npop)
        ub = self.ub
    
        # Create linear constraints
        A = pl.eye(3*npop)
        b = pl.zeros(3*npop)
        for i in range(npop-1):
            # setting constraints on z0
            A[i, i+1] = -1
            # setting constraints on pop width
            A[i + npop, i] = 1
            A[i + npop, i + 1] = -1
            A[i + npop, i + npop + 1] = 1
            # no additional constraints on slope
        # Treat the last pop width separately
        A[2*npop-1, npop - 2] = 1
        A[2*npop-1, npop - 1] = -1
        
        b[npop-1] = ub[npop-1]    
        b[2*npop:] = ub[2*npop:]
        
        return A, b

    def _lfpConstraints( self ):
        '''
        This function ...
                sets the constraints on the LFP spatial profile to be none
   
        '''
        A, b = (None, None)
        
        return A, b

    def _muaWarmup( self, *f_args ):
        '''
        This function ...
                doesn't really serve a purpose
        Aguments
        --------
    
        Keyword arguments
        -----------------
        '''
        return f_args

    def _lfpWarmup( self, rmat, kernel):
        '''
        This function ...
                reshapes the firing rate matrix from 3D to 2D
    
        Arguments:
                rmat:     firing rate
                kernel:   type of kernel chosen
                
        Output:
                rmat:     firing rate
                kernel:   type of kernel chosen
        '''
        
        if rmat.ndim==3:
            rmat = self._reshapeMat( rmat )
            
        return rmat, kernel
    
    def errEval( self, lpamat, Smat, Tmat):
        '''
        This function ...
                computes the relative mean square error between the predicted MUA or LFP
    
        Arguments:
                lpamat: the signal, either MUA or LFP
                Smat:   firing rate if the signal is MUA, postsynaptic temporal profile (firing rate convolved with the                                 temporal kernel) if the signal is LFP
                Tmat:   spatial profile. M if MUA signal, L if LFP signal
                
        Output:
                err:    relative mean square error
        '''
                
        lpamat_est = pl.dot(Smat, Tmat)
        lpamat_diff = lpamat - lpamat_est

        err = pl.mean(lpamat_diff**2)/pl.mean(lpamat**2)
        #err = pl.mean(lpamat_diff[:,200:450]**2)/pl.mean(lpamat[:,200:450]**2)

        return err

def _createLmat(  ):
    '''
    This function ...
            not used
    Aguments
    --------
    
    Keyword arguments
    -----------------
    '''
    pass
    

def _createRmat( h_list, rmat, nstim, ntime ):
    '''
    This function ...
            computes the postsynaptic temporal profile by convolving the firing rate with the temporal profile
    
    Arguments:
            h_list: the temporal kernel
            rmat: firing rates
            nstim: number of stimuli
            ntime: length of recording
                
    Output:
            Rmat: postsynaptic temporal profile
    '''
    
    nsplit = len(h_list)
    npop = int(rmat.shape[0])
    
    #print('h_list: ', h_list)
    
    #print(rmat.shape, nsplit, nstim, ntime, npop)
    
    
    if nsplit == npop:
        Rmat = pl.zeros((npop, nstim*ntime))
        for istim in range(nstim):
            for ipop in range(npop):
                rvec = rmat[ipop, istim*ntime:(istim+1)*ntime]
                h = h_list[ipop]
                #print('h ', h, 'rvec: ', rvec)
                tmp_vec= pl.convolve(rvec, h)
                tmp_vec=tmp_vec[:ntime]
                Rmat[ipop,istim*ntime:(istim+1)*ntime]=tmp_vec
    else: # single common kernel (original implementation)
        Rmat = pl.zeros((nsplit*npop, nstim*ntime))
        for istim in range(nstim):
            for isplit in range(nsplit):
                for ipop in range(npop):
                    rvec = rmat[ipop, istim*ntime:(istim+1)*ntime]
                    h = h_list[isplit]
                    #print('h ', h, 'rvec: ', rvec)
                    tmp_vec= pl.convolve(rvec, h)
                    tmp_vec=tmp_vec[:ntime]
                    Rmat[ipop+isplit*npop,\
                             istim*ntime:(istim+1)*ntime]=tmp_vec

    return Rmat

    
def _createMmat( M_params, el_coords ):
    '''
    This function ...
            computes the spatial profiles, M
    
    Arguments
            M_params:  parameters of the spatial profiles
            el_coords: positions of the electrodes
    
    Output:
            Mmat:      spatial profiles, M
    '''
    
    # Assume now that all measures are real measures in mm
    M_params = pl.asarray(M_params)
        
    nchan = el_coords.size
    
    npop = int(M_params.shape[0]/3)
    z0=M_params[:npop]
    a=M_params[npop:2*npop]
    b=M_params[2*npop:]                
        
    # Create the Mmat matrix
    Mmat = pl.zeros((nchan, npop))
            
    # Update the basis matrix, Mmat
    for ipop in range(npop):
        # tmpvec is the distance between z0[ipop] and all electrode contacts
        tmpvec = abs(el_coords - z0[ipop])
        # Set all entries that are closer than a[ipop] to 1
        Mmat[np.where(tmpvec < a[ipop])[0].tolist(), ipop] = 1 
        # Find the entries that are on the olique part of the profile and set
        # the right value
        tmpvec = tmpvec - a[ipop]
        cond1 = np.where(tmpvec >= 0)[0].tolist()
        cond2 = np.where(tmpvec < b[ipop])[0].tolist()
        isect = list(filter(lambda x: x in cond1, cond2))
        Mmat[isect, int(ipop)] = 1 - 1 / b[int(ipop)] * (tmpvec[isect])
        
    return Mmat

def _create_rmat( Mmat, muamat, muavar, rneg_factor):
    '''
    This function ...
            compoutes the firing rates
    
    Arguments
            Mmat:        spatial profiles
            muamat:      MUA signal
            muavar:      variance of the MUA signal
            rneg_factor: the limit at which you rectify
    
    Output:
            rmat:        firing rates
    '''
    
    # Estimate rmat based on pseudoinverse of Mmat
    rmat=pl.dot(pl.linalg.pinv(Mmat), muamat)
    # Rectification of rmat
    rmat[np.where(rmat<-rneg_factor*pl.sqrt(muavar))[0].tolist()]= \
        -rneg_factor*pl.sqrt(muavar)
    
    return rmat

def _singleExp(x, dt):
    '''
    This function ...
            creates the temporal kernel as defined in Laminar Population Analysis..., Einevoll et al. (2007)
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    tau=float(x[0])
    Delta=float(x[1])
    #amp = float(x[2])
    t = pl.arange(0, (10*tau), dt)
    h = pl.zeros(len(t))
    
    h = 1/tau*pl.exp(-(t-Delta)/tau)
    h[np.where(t<Delta)[0].tolist()]=0
    
    #print('tau =', np.round(tau,2), 'Delta =', np.round(Delta, 2))

    return [h]

def _singleAlpha(x, dt):
    '''
    This function ...
            creates a temporal kernel
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    
    tau = float(x[0])
    Delta = float(x[1])
    #amp = float(x[2])
    t = pl.arange(0, (10*tau), dt)
    h = pl.zeros(len(t))
    h = (t-Delta)/tau**2*pl.exp(-(t-Delta)/tau)
    h[np.where(t<Delta)[0].tolist()] = 0

    return [h]

def _doubleExp(x, dt):
    '''
    This function ...
            creates _two_ temporal kernels as defined in Einevoll (2007)
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    h1 = _singleExp(x[:2], dt)
    h2 = _singleExp(x[2:], dt)
    h = [h1[0], h2[0]]

    return h

def _tripleExp(x, dt):
    '''
    This function ...
            creates _three_ temporal kernels as defined in Einevoll (2007)
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    h1 = _singleExp(x[:2], dt)
    h2 = _singleExp(x[2:4], dt)
    h3 = _singleExp(x[4:6], dt)
    #h3 = _singleExp(x[4:6], dt)
    h = [h1[0], h2[0], h3[0]]

    return h

def _uniqueKernelsExp(self,x,dt,npop):
    '''
    This function ...
            creates temporal kernels unique to each population
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    #print('npop: ', npop)
    h = [[]]*npop
    #print('Nparams: ', len(x) // npop, 'npop: ', npop)
    tau_temp_list = []
    delta_temp_list = []
    for ipop in range(npop):
        h_ipop = _singleExp(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)], dt)
        
        tau_temp_list.append(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)][0])
        delta_temp_list.append(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)][1])
        #print('h_ipop: ', h_ipop)
        h[ipop] = h_ipop[0]
        
    self.tau = tau_temp_list
    self.delta = delta_temp_list

    return h

def _uniqueKernelsAlpha(self,x,dt,npop):
    '''
    This function ...
            creates temporal kernels unique to each population
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    #print('npop: ', npop)
    h = [[]]*npop
    tau_temp_list = []
    delta_temp_list = []
    for ipop in range(npop):
        h_ipop = _singleAlpha(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)], dt)
        
        tau_temp_list.append(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)][0])
        delta_temp_list.append(x[ipop*(len(x) // npop):(ipop+1)*(len(x) // npop)][1])
        #print('h_ipop: ', h_ipop)
        h[ipop] = h_ipop[0]
        
    self.tau = tau_temp_list
    self.delta = delta_temp_list

    return h
    

def _doubleAlpha(x, dt):
    '''
    This function ...
            creates two temporal kernel on alpha form
    
    Arguments
            x: contains the time constant, tau, and the delay, delta
            dt: time step
            
    Output:
            h: temporal kernel values
    '''
    h = []
    h.append(_singleAlpha(x[:2], dt))
    h.append(_singleAlpha(x[2:], dt))

    return h

def _gauss_kernel(x, dt):
    tau = float(x[0])
    Delta = float(x[1])
    t = pl.arange(0, (10*tau), dt)
    h = pl.zeros(len(t))
    
    h = 1/np.sqrt(2*np.pi*tau**2)*np.exp(-(t-Delta)**2/(2*tau**2))
    return h
            
class randw(object):
    def __init__( self ):
        pass
    def __call__( self ):
        pass