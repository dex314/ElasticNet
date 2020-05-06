'''
    Elastic Net Utilites
    WES.12.11.2018
'''

import numpy as np
import numpy.random as npr
import numpy.matlib as npm
from scipy.special import psi
from scipy.special import gammaln
from collections import namedtuple, Counter
import matplotlib.pyplot as plt
import pandas as pd
import dask
import operator

"""
The utils contain the following:
    Cross Validation Class using Dask for Simple Parallelization:
        fit_cv()
    Path Plot:
        path_plot(Bs, B0s, figsize=(12,8))
    Deviance (Error) Plot:
        dev_plot(dev, figsize=(12,8))
    Field Vote Plot:
        fvc_plot(Bs, min_ce_idx, param_nm, figsize=(12,8))

"""

class CrossVal(object):
    '''
    This is a cross validation method that is able to be used on any Elastic Net Model
    using only one lambda sequence and automatically parallelized with Dask.

    Initializations:
        cv_its (default = 16) = Number of cross validations.
        n_ahead (default = 4) = Number of points to validate against in each cross val
                helps to imagine this as a time series and we check these last nahead points
                against the previous points.

    Methods:
        fit_cv() - The crossvalidation scheme.

    Usage Example:
    mod = enet.ElasticNet(X, y, offset=None, x_std=False, y_std=False,
                          alpha=1.0, depth=30, tol=tols, fam='Gauss',
                          manual_lam_seq=None)
    cv = CrossVal(mod, cv_its=4, n_ahead=10)
    cv.fit_cv()

    ** For plotting and assuming mod = Model **
    _ = dev_plot(mod.dev_m, figsize=(6,4))
    _ = path_plot(mod.B, mod.B0, figsize=(12,8))
    coefs_df = mod.field_vote_plot(mod.B, mod.min_errlm_idx, list(mod.param_nm), mod.min_indices or None, figsize=(6,4))

    '''
    def __init__(self, model, cv_its=16, n_ahead=4):
        assert cv_its > 1, "Cross Vals must be > 1!"

        self.cv_its = cv_its
        self.n_ahead = n_ahead
        self.model = model

    def sigmoid(self,z):
        '''sigmoid function stand alone'''
        return 1.0/(1.0+np.exp(-z))

    def fit_cv(self):
        ''' fit call for the cross validation.
            the CV is specific for time series in keeping
            the structure intact where cv_its is the number
            of crossvalidations and n_ahead is the number of
            points to use in the validation.
            the cv slides across the time series leaving out CV2
            number of points at the beginning / end until
            reaching the end of the data set
        '''
        ## INITIALIZE
        X = self.model.x
        y = self.model.y
        offset = self.model.offset
        family = self.model.family
        tol = self.model.tol
        alpha = self.model.alpha
        depth = self.model.depth
        manual_lam_seq = self.model.manual_lam_seq
        lams = self.model.lams
        random_state = self.model.random_state

        mx, nx = np.shape(X)
        my, ny = np.shape(y)
        p = self.cv_its
        nah = self.n_ahead

        Bs = np.zeros((p, nx, depth))
        B0s = np.zeros((p, depth))
        Ks = np.zeros((p, depth))
        mod_err = np.zeros((p, depth))
        mindices = []

        ## dasked
        print('In Cross Validation #')
        results = []
        for i in range(p):
            res = dask.delayed(self.cv_par)(X, y, offset, family, tol, alpha, depth,
                                            self.model, mx, nx, my ,ny, i, p, nah)
            results.append((i+1,res))
        results = dask.compute(*results)

        '''
        The appended results i,res definitiely works better.
        The bottom results print wasnt correct either.
        The way this, the paths work out but there is lower error than the normal version.
        '''

        ## change the results to a dict and sort it by i
        print("Not Sorted")
        res_dict = sorted(dict(results).items(), key=operator.itemgetter(0))
        print("Sorted :")
        print([res_dict[jj][0] for jj in range(len(res_dict))])
        ## unpack the sorted results
        for j in range(len(res_dict)):
            Bs_r, B0s_r, Ks_r, min_ind_r, mod_err_r = res_dict[j][1]
            Bs[j,:,:] = Bs_r
            B0s[j,:] = B0s_r
            Ks[j,:] = Ks_r
            mod_err[j,:] = mod_err_r
            mindices.append(min_ind_r)

        rowmi, colmi = np.where(mod_err == np.nanmin(mod_err))
        rowm, colm = rowmi[0], colmi[0]
        beta_cnt_chk = np.sum( Bs[:, rowm] != 0)
        while beta_cnt_chk < 2 and rowm < depth-1:
            self.min_ce_idx_note = 'Min lambda error has no betas - moving forward until there are at least 2.'
            rowm += 1
            beta_cnt_chk = np.sum( Bs[:, rowm] != 0)
        min_ce_idx = rowm

        self.param_nm = self.model.param_nm
        self.B = Bs
        self.B0 = B0s
        self.lams = lams
        self.min_cvlam_idx = min_ce_idx
        self.K = Ks
        self.model_errors = mod_err
        self.min_indices = mindices

    ## nah for Parallelization
    def cv_par(self, X, y, offset, family, tol, alpha, depth,
                modTemp, mx, nx, my ,ny, i, p, nah):
        print(str(i+1)+' ',end='')
        mpi = my-p+i
        trn0 = int(i)
        trnF = int(mpi-nah)
        val0 = int(mpi-nah)
        valF = int(mpi+1)

        xt = X[trn0:trnF,:];  yt = y[trn0:trnF,:];  ot = offset[trn0:trnF,:]
        xv = X[val0:valF,:];  yv = y[val0:valF,:];  ov = offset[val0:valF,:]

        '''STACK THE CV SETS'''
        ## the Kronecker stack sorts out for a set of Betas per y vector
        ## the tile stack sorts out one set of Betas for all y vectors
        ## it makes zero sense to perform a kornecker stack when its the
        ## same as doing each item in the y variables separately
        xts = npm.repmat(xt, ny, 1)
        yts = np.reshape(yt,(np.shape(yt)[0]*np.shape(yt)[1],1),order='F')
        xvs = npm.repmat(xv, ny, 1)
        yvs = np.reshape(yv,(np.shape(yv)[0]*np.shape(yv)[1],1),order='F')

        ots = np.reshape(ot,(np.shape(ot)[0]*np.shape(ot)[1],1),order='F')
        ovs = np.reshape(ov,(np.shape(ov)[0]*np.shape(ov)[1],1),order='F')

        ## MODEL TEMP=======================================================
        # modTemp = self.model
        ## RESPECIFY THE TARGETS BASED ON THE CV METHODS HERE
        modTemp.x = xts
        modTemp.y = yts
        modTemp.offset = ots
        ## FIT THE NEW MODEL (DIFFS CAN BE CONFIRMED IN THE nah PLOTS)
        modTemp.fit()
        ## =================================================================
        kcv = np.array(modTemp.K).ravel()
        ## These are validation errors, modTemp is train errors
        yvm = np.tile(yvs,(1,depth))
        if family == 'Gauss':
            errs = np.diag(modTemp.devi_stack(xvs, yvm, modTemp.B0.ravel(),
                                             modTemp.B, kcv, ovs, fam=family))
        else:
            errs = (modTemp.devi_stack(xvs, yvs,  modTemp.B0.ravel(),
                                      modTemp.B, kcv, ovs, fam=family)).ravel()
        colm = np.where(errs == np.nanmin(errs))[0]
        # betaCntChk = np.sum( modTemp.B[:, colm] != 0)
        # while betaCntChk < 2 and colm < depth-1:
        #     colm += 1
        #     betaCntChk = np.sum(modTemp.B[:,colm] != 0)
        return modTemp.B, modTemp.B0, kcv, colm, errs


#%% PLOTTING FUNCTIONS=========================================================
#%% Plot Variable Paths and Error Over Cross Val===============================
def path_plot(Bs, B0s, figsize=(12,8)):
    ''' a function for plotting the Beta and Beta_0 paths
        _ = path_plot(model.B, model.B0, figsize)
    '''
    if len(Bs.shape) > 2:
        r_n = np.floor(len(Bs)/4)
        if r_n == 0 or r_n == 'inf':    r_n = 1;
        c_n = np.ceil(len(Bs)/r_n)
        if c_n == 0 or c_n == 'inf':    c_n = 1;

        if r_n>1 or c_n>1:
            fp, ax_p = plt.subplots(int(r_n), int(c_n), sharex=True, sharey=True, figsize=figsize)
            ax_p = ax_p.ravel()
            for i in range(len(Bs)):
                ax_p[i].plot(Bs[i][:,:].T)
                plt.suptitle('Lasso Parameters')

            fp0, ax_b0 = plt.subplots(int(r_n), int(c_n), sharex=True, sharey=True, figsize=figsize)
            ax_b0 = ax_b0.ravel()
            for i in range(len(B0s)):
                ax_b0[i].plot(B0s[i].T, 'go')
                plt.suptitle('Intercept Convergence')
    else:
        f1, ax_c = plt.subplots(1,2,figsize=figsize)
        ax_c = ax_c.ravel()
        ax_c[0].plot(Bs.T)
        ax_c[0].set_title('Betas')
        ax_c[1].plot(B0s.T, 'go')
        ax_c[1].set_title('Intercept')

#%% Nahead AND LAMS PLOT=======================================================
def err_plot(dev, figsize=(12,8)):
    ''' a function for plotting the deviance or error
        _ = err_plot(model.model_errors.T, figsize)
    '''
    f, axe = plt.subplots(1,2, sharex=False, sharey=False, figsize=figsize)
    axe = axe.ravel()
    ##Transposing Need??????
    for j in range(2):
        if j == 0:
            de = dev.T; xlabs='Lambda Depth'
        else:
            de = dev; xlabs='CV Depth'

        lc_mn = []; lc_std = [];
        for i in range(np.shape(de)[0]):
            lc_mn.append(np.mean(de[i,:], axis=0))
            lc_std.append(np.std(de[i,:], axis=0))

        yercv = [np.array(lc_std)[:], 2.0*np.array(lc_std)[:]]
        axe[j].errorbar(range(len(lc_mn)), np.array(lc_mn)[:], yerr=yercv, c='r',
                                                    marker='o', ms=4, mew=1.5, mec='k')
        axe[j].set_xlabel(str(xlabs))
    plt.suptitle('Cross Validation Deviance (Error)')

#%% NEW FIELD VOTE PLOT ======================================================
def field_vote_plot(Bs, min_ce_idx, param_name, min_indices=None, figsize=(12,8)):
    ''' New field vote plot.
        coefficient_dataframe = field_vote_plot(model.Bs, model.min_cvlam_idx, model.param_nm, cv.min_indices, figsize)
    '''
    var_counts = []
    var_vals = []
    if min_indices is not None: ##for cross val method
        cv_n, var_n, lam_n = Bs.shape
        for i in range(cv_n):
            minl = min_indices[i][0]
            midx_B = pd.DataFrame(Bs[i, :, minl].T, index=param_name, columns=['coef_val'])
            for c in list(midx_B[np.abs(midx_B.values) >= 1e-4].index.values):
                var_counts.append(c)
                var_vals.append([c,midx_B.loc[c][0]])
    else: ##single fit
        var_n, lam_n = Bs.shape
        midx_B = pd.DataFrame(Bs[:, min_ce_idx].T, index=param_name, columns=['coef_val'])
        for c in list(midx_B[np.abs(midx_B.values) >= 1e-4].index.values):
            var_counts.append(c)
            var_vals.append([c,midx_B.loc[c][0]])
    ## count up the instances over the cvs
    coef_c = []
    for key, ix in Counter(var_counts).items():
        coef_c.append([key,ix])
    ## change ot data frames for easy concatenation
    coef_cdf = pd.DataFrame(coef_c, columns=['var_name','count']).set_index('var_name')
    coef_vdf = pd.DataFrame(var_vals, columns=['var_name','var_val_mean']).set_index('var_name').groupby('var_name').mean()
    full = pd.concat([coef_cdf, coef_vdf], axis=1, join='outer')
    full = full.loc[(full!=0).any(axis=1)]
    full.columns = ['votes', 'vals']
    fulls = full.sort_values(by='votes',ascending=False)

    color2 = iter(plt.cm.rainbow(np.linspace(0,1,len(fulls))))
    f, axs = plt.subplots(1,2,sharey=True,figsize=figsize)
    axs = axs.ravel()
    for j in range(len(fulls)):
        col2 = next(color2)
        axs[0].barh(j,fulls.iloc[j,0], color=col2, align='center')
        axs[0].set_title('Field Votes'), axs[0].set_xlabel('Cross Vals')
        axs[1].barh(j,fulls.iloc[j,1], color=col2, align='center')
        axs[1].set_title('Variable Importance'), axs[1].set_xlabel('Beta Value')
        axs[1].axvline(0,color='black',linewidth=0.5)
    plt.yticks(np.arange(len(fulls.index)), fulls.index, fontsize=9)

    return full

def cv_graph(x, cv_its, n_ahead, figsize=(12,6)):
    ''' Function to plot the Cross Validation Scheme '''
    m = len(x)
    xidx = x.index
    df = pd.DataFrame(index = xidx)
    for i in range(cv_its):
        mpi = m - cv_its + i
        trn0 = int(i)
        trnF = int(mpi-n_ahead)
        val0 = int(mpi-n_ahead+1)
        valF = int(mpi)
        df['CV_{}'.format(i+1)] = 0
        df['CV_{}'.format(i+1)] = 1*((df.index>=xidx[trn0])&(df.index<=xidx[trnF])) + \
                -1*((df.index>=xidx[val0])&(df.index<=xidx[valF]))
    
    df.index = df.index.date
    fig, ax=plt.subplots(1,1,figsize=figsize)
    sns.heatmap(df.T, cmap='coolwarm', linewidth=1, cbar=False, ax=ax);
    ax.set_title("Cross Validation Graph", fontsize=16);
    ax.set_xlabel("Red = Train | Blue = Val | Gray = Unused", fontsize=16);
    plt.xticks(rotation=35);
    return df


    
    
    
    
