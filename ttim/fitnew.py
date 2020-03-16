class CalibrateLmfit:
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