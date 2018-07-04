import numpy as np
import pandas as pd
from scipy.optimize import least_squares

class Calibrate:
    
    def __init__(self, model):
        self.model = model
        self.parameters = pd.DataFrame(columns=['initial', 'optimal', 'pmin', 'pmax', 'std', 'perc_std'])
        self.seriesdict = {}
        
    def set_parameter(self, name=None, parameter=None, layer=0, initial=0, pmin=-np.inf, pmax=np.inf):
        """
        if name is 'kaq#' or 'Saq#', no parameter of layer needs to be provided
        otherwise, parameter needs to be an array and layer needs to be specified
        """
        if parameter is not None:
            assert isinstance(parameter, np.ndarray), "Error: parameter needs to be numpy array"
            p = parameter[layer:layer + 1]
        elif isinstance(name, str):
            # Set, kaq, Saq
            if name[:3] == 'kaq':
                layer = int(name[3:])
                p = self.model.aq.kaq[layer:layer + 1]
            elif name[:3] == 'Saq':
                layer = int(name[3:])
                p = self.model.aq.Saq[layer:layer + 1]
            elif name[0] == 'c':
                layer = int(name[1:])
                p = self.model.aq.c[layer:layer + 1]
            # TO DO: set c, Sll
        else:
            print('parameter name not recognized or no parameter reference supplied')
            return
        self.parameters.loc[name] = {'initial':initial, 'pmin':pmin, 'pmax':pmax, 'optimal':p[:], 'std':None, 'perc_std':None}
        #self.parametersdict[name] = p
        
    def series(self, name, x, y, layer, t, h):
        s = Series(x, y, layer, t, h)
        self.seriesdict[name] = s
        
    def residuals(self, p):
        print('.', end='')
        # set the values of the variables
        for i, k in enumerate(self.parameters.index):
            self.parameters.loc[k, 'optimal'][:] = p[i]  # [:] needed to do set value in array
        self.model.solve(silent=True)
        rv = np.empty(0)
        for key in self.seriesdict:
            s = self.seriesdict[key]
            h = self.model.head(s.x, s.y, s.t, layers=s.layer)
            rv = np.append(rv, s.h - h)
        return rv
    
    def fit(self, report=True):
        self.fitresult = least_squares(self.residuals, self.parameters.initial.values,
                                        bounds=(self.parameters.pmin.values, self.parameters.pmax.values),
                                        method='trf', diff_step=1e-5, xtol=1e-8)
        # Call residuals to specify optimal values for model
        res = self.residuals(self.fitresult.x)
        print('')
        nparam = len(self.fitresult.x)
        H = self.fitresult.jac.T @ self.fitresult.jac
        sigsq = np.var(res, ddof=nparam)
        self.covmat = np.linalg.inv(H) * sigsq 
        self.sig = np.sqrt(np.diag(self.covmat))
        D = np.diag(1 / self.sig)
        self.cormat = D @ self.covmat @ D
        self.parameters['std'] = self.sig
        self.parameters['perc_std'] = self.sig / self.parameters['optimal'] * 100
        if report:
            print(self.parameters)
            print(self.sig)
            print(self.covmat)
            print(self.cormat)
            
            
class CalibrateOld:
    def __init__(self, model):
        from lmfit import Parameters, minimize, fit_report
        self.model = model
        self.lmfitparams = Parameters()
        self.parameterdict = {}
        self.seriesdict = {}
    def parameter(self, name, par=None, layer=0, initial=0, pmin=None, pmax=None, vary=True):
        if par is not None:
            assert type(par) == np.ndarray, "Error: par needs to be array"
            p = par[layer:layer + 1]
        else:
            if name[:3] == 'kaq':
                layer = int(name[3:])
                p = self.model.aq.kaq[layer:layer + 1]
            elif name[:3] == 'Saq':
                layer = int(name[3:])
                p = self.model.aq.Saq[layer:layer + 1]
            else:
                print('parameter name not recognized or no par reference supplied')
                return
        self.lmfitparams.add(name, value=initial, min=pmin, max=pmax, vary=vary)
        self.parameterdict[name] = p
    def series(self, name, x, y, layer, t, h):
        s = Series(x, y, layer, t, h)
        self.seriesdict[name] = s
    def residuals(self, p):
        # p is lmfit.Parameters object
        print('.', end='')
        vals = p.valuesdict()
        for k in vals:
            self.parameterdict[k][:] = vals[k]  # [:] needed to do set value in array
            # do something else when it is the storage coefficient
            # this needs to be replaced when Saq computation is moved to initialize
            if len(k) > 3:
                if k[:3] == 'Saq':  
                    layer = int(k[3:])
                    if layer == 0:
                        if self.model.aq.phreatictop:
                            self.parameterdict[k][:] = vals[k]
                        else:
                            self.parameterdict[k][:] = vals[k] * self.model.aq.Haq[0]
                    else:
                        self.parameterdict[k][:] = vals[k] * self.model.aq.Haq[layer]
        self.model.solve(silent=True)
        rv = np.empty(0)
        for key in self.seriesdict:
            s = self.seriesdict[key]
            h = self.model.head(s.x, s.y, s.t, layers=s.layer)
            rv = np.append(rv, s.h - h)
        return rv
    def fit(self, report=True):
        self.fitresult = minimize(self.residuals, self.lmfitparams, epsfcn=1e-4)
        if report:
            print(fit_report(self.fitresult))
    
class Series:
    def __init__(self, x, y, layer, t, h):
        self.x = x
        self.y = y
        self.layer = layer
        self.t = t
        self.h = h


