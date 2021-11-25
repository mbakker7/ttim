import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import re
import lmfit

class Calibrate:
    
    def __init__(self, model):
        """initialize Calibration class
        
        Parameters
        ----------
        model : ttim.Model
            model to calibrate
        
        """

        self.model = model
        self.parameters = pd.DataFrame(columns=[
            'optimal', 'std', 'perc_std', 'pmin', 'pmax', 'initial', 'parray'])
        self.seriesdict = {}
        self.seriesinwelldict = {}
        
    def set_parameter(self, name=None, initial=0, pmin=-np.inf, pmax=np.inf):
        """set parameter to be optimized
        
        Parameters
        ----------
        name : str
            parameter name, can include layer information. 
            name can be 'kaq', 'Saq' or 'c'. A number after the parameter 
            name denotes the layer number, i.e. 'kaq0' refers to the hydraulic 
            conductivity of layer 0. 
            name also supports layer ranges, entered by adding a '_' and a
            layer number, i.e. 'kaq0_3' denotes conductivity for layers 0 up to 
            and including 3.
        initial : np.float, optional
            initial value for the parameter (the default is 0)
        pmin : np.float, optional
            lower bound for parameter value (the default is -np.inf)
        pmax : np.float, optional
            upper bound for paramater value (the default is np.inf)
        
        """

        assert type(name) == str, "Error: name must be string"
        # find numbers in name str for support layer ranges
        layers_from_name = re.findall(r'\d+', name)
        p = None
        if "_" in name:
            fromlay, tolay = [np.int(i) for i in layers_from_name]
            if name[:3] == 'kaq':
                p = self.model.aq.kaq[fromlay:tolay+1]
            elif name[:3] == 'Saq':
                p = self.model.aq.Saq[fromlay:tolay+1]
            elif name[0] == 'c':
                p = self.model.aq.c[fromlay:tolay+1]
            # TODO: set Sll
        else:
            layer = np.int(layers_from_name[0])
            # Set, kaq, Saq, c
            if name[:3] == 'kaq':
                p = self.model.aq.kaq[layer:layer + 1]
            elif name[:3] == 'Saq':
                p = self.model.aq.Saq[layer:layer + 1]
            elif name[0] == 'c':
                p = self.model.aq.c[layer:layer + 1]
            # TODO: set Sll
        if p is None:  # no parameter set
            print('parameter name not recognized or no parameter ref supplied')
            return
        self.parameters.loc[name] = {'optimal':initial, 'std':None, 
                                     'perc_std':None, 'pmin':pmin, 'pmax':pmax, 
                                     'initial':initial, 'parray':p[:]}
        
    def set_parameter_by_reference(self, name=None, parameter=None, initial=0, 
                                   pmin=-np.inf, pmax=np.inf):
        """set parameter to be optimized
        
        Parameters
        ----------
        name : str
            parameter name
        parameter : np.array
            array reference containing the parameter to be optimized. must be 
            specified as reference, i.e. w.rc[0:]  
        initial : np.float, optional
            initial value for the parameter (the default is 0)
        pmin : np.float, optional
            lower bound for parameter value (the default is -np.inf)
        pmax : np.float, optional
            upper bound for paramater value (the default is np.inf)
        
        """
        assert type(name) == str, "Error: name must be string"
        if parameter is not None:
            assert isinstance(parameter, np.ndarray), \
                "Error: parameter needs to be numpy array"
            p = parameter
        self.parameters.loc[name] = {'optimal':initial, 'std':None, 
                                     'perc_std':None, 'pmin':pmin, 'pmax':pmax, 
                                     'initial':initial, 'parray':p[:]}
        
    def series(self, name, x, y, layer, t, h):
        """method to add observations to Calibration object
        
        Parameters
        ----------
        name : str
            name of series
        x : np.float
            x-coordinate
        y : np.float
            y-coordinate
        layer : int
            layer number, 0-indexed
        t : np.array
            array containing timestamps of timeseries
        h : np.array
            array containing timeseries values, i.e. head observations
        
        """

        s = Series(x, y, layer, t, h)
        self.seriesdict[name] = s
        
    def seriesinwell(self, name, element, t, h):
        """method to add observations to Calibration object
        
        Parameters
        ----------
        name : str
            name of series
        element: element object with headinside function
        t : np.array
            array containing timestamps of timeseries
        h : np.array
            array containing timeseries values, i.e. head observations
        
        """

        e = SeriesInWell(element, t, h)
        self.seriesinwelldict[name] = e
        
    def residuals(self, p, printdot=False):
        """method to calculate residuals given certain parameters
        
        Parameters
        ----------
        p : np.array
            array containing parameter values
        printdot : bool, optional
            print dot for each function call
        
        Returns
        -------
        np.array
            array containing all residuals
        
        """

        if printdot:
            print('.', end='')
        # set the values of the variables
        
        if printdot == 7:
            print(p)
        
        for i, k in enumerate(self.parameters.index):
            # [:] needed to do set value in array
            self.parameters.loc[k, 'parray'][:] = p[i]  
            
        self.model.solve(silent=True)
        
        rv = np.empty(0)
        for key in self.seriesdict:
            s = self.seriesdict[key]
            h = self.model.head(s.x, s.y, s.t, layers=s.layer)
            rv = np.append(rv, s.h - h)
        for key in self.seriesinwelldict:
            s = self.seriesinwelldict[key]
            h = s.element.headinside(s.t)[0]
            rv = np.append(rv, s.h - h)
        return rv
    
    def residuals_lmfit(self, lmfitparams, printdot=False):
        vals = lmfitparams.valuesdict()
        p = np.array([vals[k] for k in self.parameters.index])
        #p = np.array([vals[k] for k in vals])
        return self.residuals(p, printdot)
    
    def fit_least_squares(self, report=True, diff_step=1e-4, xtol=1e-8, 
                          method='lm'):
        self.fitresult = least_squares(
            self.residuals, self.parameters.initial.values, args=(True,), 
            bounds=(self.parameters.pmin.values, self.parameters.pmax.values),
            method=method, diff_step=diff_step, xtol=xtol, x_scale="jac")
        print('', flush=True)
        # Call residuals to specify optimal values for model
        res = self.residuals(self.fitresult.x)
        for ipar in self.parameters.index:
            self.parameters.loc[ipar, 'optimal'] = \
                self.parameters.loc[ipar, 'parray'][0]
        nparam = len(self.fitresult.x)
        H = self.fitresult.jac.T @ self.fitresult.jac
        sigsq = np.var(res, ddof=nparam)
        self.covmat = np.linalg.inv(H) * sigsq 
        self.sig = np.sqrt(np.diag(self.covmat))
        D = np.diag(1 / self.sig)
        self.cormat = D @ self.covmat @ D
        self.parameters['std'] = self.sig
        self.parameters['perc_std'] = self.sig / \
                                      self.parameters['optimal'] * 100
        if report:
            print(self.parameters)
            print(self.sig)
            print(self.covmat)
            print(self.cormat)
            
    def fit_lmfit(self, report=True, printdot=True):
        import lmfit
        self.lmfitparams = lmfit.Parameters()
        for name in self.parameters.index:
            p = self.parameters.loc[name]
            self.lmfitparams.add(name, value=p['initial'], min=p['pmin'], 
                                 max=p['pmax'])
        fit_kws = {"epsfcn": 1e-4}
        self.fitresult = lmfit.minimize(self.residuals_lmfit, self.lmfitparams, 
                                        method="leastsq", 
                                        kws={"printdot":printdot}, **fit_kws)
        print('', flush=True)
        print(self.fitresult.message)
        if self.fitresult.success:
            for name in self.parameters.index:
                self.parameters.loc[name, 'optimal'] = \
                    self.fitresult.params.valuesdict()[name]
            if hasattr(self.fitresult, 'covar'):
                self.parameters['std'] = np.sqrt(np.diag(self.fitresult.covar))
                self.parameters['perc_std'] = 100 * self.parameters['std'] / \
                                              np.abs(self.parameters['optimal'])
            else:
                self.parameters['std'] = np.nan
                self.parameters['perc_std'] = np.nan
        if report:
            print(lmfit.fit_report(self.fitresult))
            
    def fit(self, report=True, printdot=True):
        # current default fitting routine
        return self.fit_lmfit(report, printdot)
            
    def rmse(self):
        """calculate root-mean-squared-error
        
        Returns
        -------
        np.float
            return rmse value
        """

        r = self.residuals(self.parameters['optimal'].values)
        return np.sqrt(np.mean(r ** 2))
    
class Series:
    def __init__(self, x, y, layer, t, h):
        self.x = x
        self.y = y
        self.layer = layer
        self.t = t
        self.h = h
        
class SeriesInWell:
    def __init__(self, element, t, h):
        self.element = element
        self.t = t
        self.h = h


