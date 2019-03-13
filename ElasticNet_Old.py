'''
WES.2018.03.01
'''

import numpy as np
import numpy.random as npr
from scipy.special import psi, gammaln
from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler

#%%
class ElasticNet(object):
    '''
    This is a single use Elastic Net method for solving a Linear Equation.
    It can be used for a Gaussian, Poisson, Negative Binomial, or Logit distributions.

    Calling the fit method with no inputs allows you to develop your own cross validation method if needed.

    Initializations:
        x = feature variable(s)
        y = target variable(s)
        offset (default = None) = offset
        alpha (default=1) = regularization term
            1.0 = full Lasso regression
            0.0 = full Ridge regression
        depth (default=20) = depth of lambda to go to (full path is 100)
        tol (defalut = 1e-4) = tolerance specification for the beta convergence
        x_std (default = False) = standardize x
        y_std (default = False) = standardize y
        family = family of functions (default is NegBin)
            The other available methods are 'Gauss', 'Poisson', and 'Logit'.
        manual_lam_seq (default is None)
            For providing a manual lambda sequence. Must pass an array.
            Resets the depth to len(manual_lam_seq)-1!!!

    Methods:
        lam_seq_gen(x, y, offset=1, alpha=1, nlen=20):
            takes the passed x and y and develops the lambda sequence
        cost(x, y, b0, b, k, lam, offset=1, alpha=1, fam):
            cost function for the optimization
            not utilized in the coord descent but available here for testing
        cord(x, y, b0_init, b_init, lam, k=1, alpha=1, nullDev=1, tol=1e-4, fam='NegBin', offset=1):
            coordinate descent for optimization
        disp_est(,x,y,b0,b,offset=1,k=1):
            dispersion estimate
        devi(x, y, b0, b, k=1, offset=1.0, fam='NegBin'):
            deviance
        devi_stack(x, y, b0, b, k=1, offset=1, fam='NegBin'):
            deviance for stacked errors
        sigmoid(z):
            sigmoid function
        nbd_grad(x, y, b0, b, offset=1, k=1):
            NegBin Gradient
        fit():
            Fits the model.

    Usage Example:
        from ___.___ import ElasticNet as enet
        from enetUtils import *
        lags = 1
        xL = lagDf(xData, lags) #potential call to lag a data frame
        mod = enet.ElasticNet(xL, yB, offset=None, x_std=True, y_std=False,
                              alpha=1.0, depth=20, tol=tols, fam='Gauss',
                              manual_lam_seq=None)
        fit = mod.fit()

    '''
    def __init__(self, x, y, offset=None, x_std=False, y_std=False,
                 alpha=1.0, depth=20, tol=1e-4, fam='NegBin',
                 manual_lam_seq=None):
        ''' initialize '''
        self.x_std = x_std
        self.y_std = y_std

        if type(x) == pd.core.frame.DataFrame:
            self.param_nm = x.columns
        else:
            self.param_nm = list(str('X'+str(x)) for x in range(x.shape[1]))

        ss = StandardScaler(with_mean=True, with_std=True)

        if x_std == True:
            self.x = ss.fit_transform(x)
        else:
            self.x = np.array(x)

        if y_std == True:
            if len(np.shape(y)) > 1:
                self.y = ss.fit_transform(y)
            else:
                #y = ss.fit_transform(y[:, None])[:, 0]
                self.y = ss.fit_transform(y.reshape(-1,1))
        else:
            self.y = np.array(y)

        if fam == 'Logit':
            self.y = np.array(np.where(y>0,1,0))
            ## this changes the target to a binary set

        if len(np.shape(self.y))>1:
            pass
        else:
            self.y = np.reshape(self.y, (len(self.y),1))

        if offset is not None:
            ##check shape or size
            if np.size(offset) == 1:
                if offset == 0:
                    self.offset = np.ones(self.y.shape)
                else:
                    self.offset = offset * np.ones(self.y.shape)
        else:
            self.offset = np.ones(self.y.shape)

        assert len(self.offset) == len(self.y), "Length of Offset != Length of y"
        self.offset = np.reshape(self.offset, (len(self.offset),1))

        self.alpha = alpha
        self.depth = depth
        self.tol = tol
        self.family = fam

        ##FORMAT LAMBDA SEQ NOW
        mx, nx = np.shape(self.x)
        my, ny = np.shape(self.y)
        if fam == 'NegBin' or fam == 'Poisson':
            b0_init = np.log(np.mean(self.y/self.offset, axis=0))
        if fam == 'Gauss':
            b0_init = np.mean(self.y,axis=0)
        if fam == 'Logit':
            b0_init = np.log(np.mean(self.y,axis=0)/(1-np.mean(self.y,axis=0)))
        ## CHECKING FOR MULTIVARIABLE TARGET
        if ny > 1:
            xstack = np.matlib.repmat(self.x, ny, 1)
            ystack = np.reshape(self.y, (my*ny,1), order='F')
            ofstack = np.reshape(self.offset, (my*ny,1), order='F')
        if ny>1:
            fStackCase = {'NegBin': ystack - np.exp(b0_init + np.log(ofstack)),
                          'Poisson': ystack - np.exp(b0_init + np.log(ofstack)),
                          'Gauss': ystack - b0_init,
                          'Logit': ystack - self.sigmoid(b0_init)}
            funstack = fStackCase[self.family]
            lams = self.lam_seq_gen(xstack, funstack, ofstack, self.alpha, 100)
        else:
            fCase = {'NegBin': self.y - np.exp(b0_init + np.log(self.offset)),
                     'Poisson': self.y - np.exp(b0_init + np.log(self.offset)),
                     'Gauss': self.y - b0_init,
                     'Logit': self.y - self.sigmoid(b0_init)}
            fun = fCase[self.family]
            lams = self.lam_seq_gen(self.x, fun, self.offset, self.alpha, 100)
        self.lams = lams

        if manual_lam_seq is None:
            pass
        else:
            manual_lam_seq = np.array(manual_lam_seq)
            if type(manual_lam_seq) != np.ndarray and type(manual_lam_seq) != list:
                raise Exception('** Manual lambdas must be a list or an numpy array and must be of length >= 2! **')
            assert len(manual_lam_seq) >= 2, "** Length of Manual Lam Seq Must Be >= 2. **"
            self.lams = manual_lam_seq.astype(float)
            self.depth = len(manual_lam_seq) - 1
            print(" ** Depth has been reset appropriately to reflect manual lambda sequence! ** ")
        self.manual_lam_seq = manual_lam_seq

    #%% LOG LAMBDA SEQUENCE FUNCTION===========================================
    def lam_seq_gen(self, x, y, offset=1, alpha=1, nlen=100):
        ''' lambda sequence generator '''
        m,n = np.shape(x)
        ## addition to assist with sizing problems coming from the offset and y
        ## if y is not already standardized
        if np.mean(np.abs( np.dot(y.T,y) - len(y) )) < 1e-2:
            pass
        else:
            y = y / np.std(y)

        if m>n:  lam_ratio = 0.0001
        else:    lam_ratio = 0.01
        lam_max = np.max( np.abs( np.dot(x.T,y) ) ) / m
        if alpha != 0:  lam_max = lam_max / alpha
        else:           lam_max = lam_max / 0.001
        lam_min = lam_ratio*lam_max
        lams_log = np.linspace(np.log(lam_max), np.log(lam_min), 100)
        lams = np.exp(np.insert(lams_log,0,-10))
        return lams

    #%% NEGATIVE BINOMIAL LASSO FUNCTION=======================================
    def fit(self):
        ''' fit call for the regression '''
        fam = self.family
        X = self.x
        y = self.y
        ofs = self.offset

        mx, nx = np.shape(X)
        my, ny = np.shape(y)

        b_init = np.zeros((nx,1))
        if fam == 'NegBin' or fam == 'Poisson':
            b0_init = np.log( np.mean(y/ofs, axis=0))
            k_init, it_dummy = self.disp_est(X, y, b0_init, b_init, ofs, 1)
            if fam == 'Poisson':
                k_init, it_dummy = 1e-5, 0
            dev = self.devi(X, y, b0_init, b_init, k_init, ofs, fam)
        if fam == 'Gauss':
            b0_init = np.mean(y,axis=0)
            k_init, it_dummy = 1e-5, 0
            dev = np.mean(self.devi(X,y,b0_init,b_init, k_init, ofs, fam))
        if fam == 'Logit':
            p0 = np.mean(y,axis=0)
            b0_init = np.log(p0/(1-p0))
            k_init, it_dummy = 1e-5, 0
            dev = self.devi(X, y, b0_init, b_init, k_init, ofs, fam)

        ## New way to intialize lambdas
        lams = self.lams

        if np.isnan(b0_init).any() == True:
            raise Exception("The value of b0 is NAN. Confirm y is NOT standardized.")

        ##Storage Containers for Variables--------------------------------------
        minL = min(self.depth, 100)
        betas = np.zeros((nx, minL))
        beta0s = np.zeros((1, minL))
        ks = np.zeros((1, minL))
        yhats = np.zeros((minL, my))
        disp_iters = np.zeros((minL,1))
        mod_err = np.zeros((minL,1))

        ##---------------------------------------------------------------------
        for j in range(minL):
            lnb1 = lams[j+1]
            lnb0 = lams[j]
            if fam == 'NegBin':
                k, disp_iter = self.disp_est(X, y, b0_init, b_init, ofs, k_init)
            else:
                k, disp_iter = 1e-5, 0

            nzb, jdum = np.nonzero( np.abs(X.T.dot(y) / mx) > self.alpha*(2.0*lnb1 - lnb0) )
            x_nzb = np.array(X[:,nzb])
            b_nzb = np.array(b_init[nzb])

            b0, b, npass = self.cord(x_nzb, y, b0_init, b_nzb, lnb1, k, self.alpha, dev/mx, self.tol, fam, ofs)

            b0_init = np.copy(b0)
            k_init = np.copy(k)
            b_init[nzb] = b[:]

            if fam == 'NegBin' or fam == 'Poisson':
                model_dev = self.devi(X,y,b0_init,b_init,k_init,ofs,fam=fam)
                r = np.divide(np.subtract(dev,model_dev),dev)
                if r > 0.9:  break
                yhat = np.exp(b0_init + X.dot(b_init) + np.log(ofs))
            if fam == 'Logit':
                model_dev = self.devi(X,y,b0_init,b_init,k_init,ofs,fam=fam)
                r = np.divide(np.subtract(dev,model_dev),dev)
                if r > 0.9:  break
                yhat = self.sigmoid(b0_init + X.dot(b_init))
            else:
                model_dev = np.mean(self.devi(X,y,b0_init,b_init,k_init,ofs,fam=fam))
                yhat = b0_init + X.dot(b_init)

            betas[:,j] = np.copy(b_init.ravel())
            beta0s[:,j] = np.copy(b0_init)
            ks[:,j] = np.copy(k_init)
            yhats[j,:] = yhat.ravel()
            disp_iters[j] = disp_iter
            mod_err[j] = model_dev

            if k_init <= 1e-4:
                self.DispersionNote = "Dispersion reached < 1e-4, consider running a Poisson."

        ## MIN OUT OF SAMPLE ERROR PREDICTION - PICKING LOWEST LAMBDA WITH AT LEAST 2 BETAS
        min_errlm_idx = np.where(mod_err == np.nanmin(mod_err))[0][0]
        betaCntChk = np.sum(betas[:,min_errlm_idx]!=0)
        while betaCntChk < 2 and min_errlm_idx < self.depth-1:
            self.min_errlm_idx_note = 'Min lambda error had no Betas - moving forward until there are at least 2.'
            min_errlm_idx += 1
            betaCntChk = np.sum(betas[:,min_errlm_idx]!=0)

        self.B = betas
        self.B0 = beta0s
        self.min_lam_idx = min_errlm_idx
        self.K = ks
        self.disp_iter = disp_iters
        self.yhat = yhats
        self.model_errors = mod_err

    #%% DISPERSION ESTIMATE FOR K==============================================
    def disp_est(self, x, y, b0, b, offset=1, k=1):
        ''' dispersion estimate calculation '''
        iters = 0
        k_old=0
        while np.abs(k-k_old) > 1e-3:
            k_old = np.copy(k)
            k = k - 0.01 / np.sqrt(len(x)+iters) * self.nbd_grad(x,y,b0,b,offset,k) ##Original
            iters += 1
            if k<0:
                k = 1e-6
                break
        return k, iters

    #%% GRADIENT - NEG BINOM===================================================
    def nbd_grad(self, x, y, b0, b, offset=1, k=1):
        ''' gradient calculation for the negative binomial model '''
        mu = np.exp(b0 + x.dot(b) + np.log(offset))
        grad = -np.sum( psi(y+1/k)*(-1/k**2) + psi(1/k)*(1/k**2) + (1/k**2)*np.log(k) - \
                (1/k**2) + (1/k**2)*np.log(1/k + mu) + (1/k**3)/(1/k + mu)  + \
                (y/(1/k + mu))*(1/k**2) )
        return grad

    #%% SIGMOID================================================================
    def sigmoid(self,z):
        ''' sigmoid function 1/(1+exp(-z)) for logit '''
        return 1.0/(1.0+np.exp(-z))

    #%% COORDINATE DESCENT - NEG BINOM=========================================
    def cord(self, x, y, b0_init, b_init, lam, k=1, alpha=1, nullDev=1, tol=1e-4, fam='NegBin', offset=1):
        ''' coordinate descent algorithm based on beta convergence '''
        m,n = np.shape(x)
        npass, tol_chk = 0, 1
        b = np.zeros((n,1))

        if fam == 'Gauss':
            w = np.ones((len(y),1))
            z = y
        if fam == 'NegBin':
            p = np.exp(b0_init + np.add(x.dot(b_init), np.log(offset)))
            s = np.divide( ((k*y+1.0)*p) , (k*p + 1.0)**2 )
            q0 =  np.divide( (k*p+1.0) , ((k*y+1.0)*p) )
            w = np.ones((len(y),1))*s
            z = b0_init + np.add(x.dot(b_init), np.subtract(y,p)*q0)
        if fam == 'Logit':
            p = self.sigmoid(b0_init + np.dot(x,b_init))
            s = np.multiply( p, (1.0-p) )
            q0 =  np.divide( (y-p) , s )
            w = np.ones((len(y),1))*s
            z =  b0_init + np.add(x.dot(b_init), q0)
        if fam == 'Poisson':
            p = np.exp(b0_init + np.add(x.dot(b_init), np.log(offset)))
            q0 =  np.divide( (y-p) , p )
            w = np.ones((len(y),1))*p
            z =  b0_init + np.add(x.dot(b_init), q0)

        while tol_chk >= tol and npass<1000:
            npass+=1
            b0 = np.dot( w.T, np.subtract(z, np.dot(x,b))) / np.sum(w)
            if x.size != 0:
                for ii in range(0,n):
                    xi = x[:,[ii]]
                    b[ii] = np.dot(xi.T, ( w*(np.subtract(z, np.dot(x,b)) - b0 + xi*b[ii]) ) )/m
                    f = np.abs(b[ii]) - alpha*lam
                    st = np.sign(b[ii]) * (np.abs(f) + f)/2.0 ## SoftThreshHolding
                    b[ii] = np.divide(st , np.add( np.dot(xi.T, (w*xi))/m , (1.0-alpha)*lam ))
            tol_chk = np.linalg.norm(np.subtract(b0+b, b0_init+b_init))
            b_init[:] = b[:]
            b0_init[:] = b0[:]

        return b0, b, npass

    #%% COST FUNC==============================================================
    ## Not really in use with the particular Coordinate Descent being used but still a resource.
    def cost(self, x, y, b0, b, lam, k=1, offset=1.0, alpha=1, fam='NegBin'):
        ''' cost function * no longer used but useful if needed '''
        m,n=np.shape(x)
        reg = lam*alpha*np.sum(np.abs(b)) + lam*(1.0-alpha)*np.linalg.norm(b)
        j = -self.devi(x,y,b0,b,k,offset,fam)/m
        return (j + reg)

    #%% DEVIANCE===============================================================
    def devi(self, x, y, b0, b, k=1, offset=1.0, fam='NegBin'):
        ''' deviance calculation for each family '''
        m,n=np.shape(x)
        if fam == 'NegBin':
            mu = np.array(np.exp(b0 + x.dot(b) + np.log(offset)), ndmin=2)
            LL = gammaln(y + 1/k) - gammaln(1/k) - gammaln(y + 1) - (y + 1/k)*np.log(1 + k*mu) + y*np.log(k) + y*np.log(mu)
            L = -2.0*np.sum(LL)
        if fam == 'Poisson':
            if offset.all() == 1.0:
                mu = np.array(np.exp(b0 + x.dot(b) + np.log(offset)), ndmin=2)
                L = -2.0*np.sum(y*mu - gammaln(y+1))
            else:
                mu = np.array(np.exp(b0 + x.dot(b)), ndmin=2)
                L = -2.0*( (y/offset).T * mu.T * (1/offset) )
        if fam == 'Gauss':
            res = np.subtract(y, x.dot(b) + b0)
            L = 0.5*np.dot(res.T,res)
        if fam == 'Logit':
            mu = np.array(self.sigmoid(b0 + x.dot(b)), ndmin=2)
            L = -2.0*np.sum( np.add( np.where(y>0, y*np.log(mu), 0), np.where(y<1, (1.0-y)*np.log(1.0-mu), 0) ))
        return (L)

    def devi_stack(self, x, y, b0, b, k=1, offset=1, fam='NegBin'):
        """ deviance calculation for the stacked multivariate target model """
        m,n=np.shape(x)
        if fam == 'NegBin':
            mu = np.exp(b0 + x.dot(b) + np.log(offset))
            LL = gammaln(y + 1/k) - gammaln(1/k) - gammaln(y + 1) - (y + 1/k)*np.log(1 + k*mu) + y*np.log(k) + y*np.log(mu)
            L = -2.0*np.sum(LL, axis=0)
        if fam == 'Poisson':
            if offset.all() == 1.0:
                mu = np.array(np.exp(b0 + x.dot(b) + np.log(offset)))
                L = -2.0*np.sum(y*mu - gammaln(y+1), axis=0)
            else:
                mu = np.array(np.exp(b0 + x.dot(b)))
                L = -2.0*( (y/offset).T * mu.T * (1/offset) )
        if fam == 'Gauss':
            LL = np.subtract(y, x.dot(b) + b0)
            L = 0.5/len(y) * LL.T.dot(LL)
        if fam == 'Logit':
            mu = self.sigmoid(b0 + x.dot(b))
            L = -2.0*np.sum( np.add( np.where(y>0, y*np.log(mu), 0),
                                    np.where(y<1, (1.0-y)*np.log(1.0-mu), 0) ), axis=0)
        return (L)


'''
*************************
END
*************************
'''
