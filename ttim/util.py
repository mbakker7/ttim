import numpy as np
import matplotlib.pyplot as plt

class PlotTtim:
    def plot(self, win=None, newfig=True, figsize=None):
        """Plot layout
        
        Parameters
        ----------
    
        win : list or tuple
            [x1, x2, y1, y2]
            
        """
        
        if newfig:
            plt.figure(figsize=figsize)
            ax1 = plt.subplot()
        else:
            fig = plt.gcf()
            ax1 = fig.axes[0]
        if ax1 is not None:
            plt.sca(ax1)
            for e in self.elementlist:
                e.plot()
            plt.axis('scaled')
            if win is not None:
                plt.axis(win)
    
    def xsection(self, x1=0, x2=1, y1=0, y2=0, npoints=100, t=1, layers=0,
                 sstart=0, color=None, lw=1, figsize=None, newfig=True,
                 legend=True):
        layers = np.atleast_1d(layers)
        if newfig:
            plt.figure(figsize=figsize)
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        s = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 ) + sstart
        h = self.headalongline(x, y, t, layers)
        nlayers, ntime, npoints = h.shape
        for i in range(nlayers):
            for j in range(ntime):
                if color is None:
                    plt.plot(s, h[i, j, :], lw=lw)
                else:
                    plt.plot(s, h[i, j, :], color, lw=lw)
        if legend:
            legendlist = ['layer ' + str(i) for i in layers]
            plt.legend(legendlist)
        #plt.draw()
        
    def contour(self, win, ngr=20, t=1, layers=0, levels=20, layout=True, 
                labels=False, decimals=0, color=None, newfig=True, 
                figsize=None, legend=True):
        """Contour plot
        
        Parameters
        ----------
    
        win : list or tuple
            [x1, x2, y1, y2]
        ngr : scalar, tuple or list
            if scalar: number of grid points in x and y direction
            if tuple or list: nx, ny, number of grid points in x and y direction
        t : scalar
            time
        layers : integer, list or array
            layers for which grid is returned
        levels : integer or array (default 20)
            levels that are contoured
        layout : boolean (default True_)
            plot layout of elements
        labels : boolean (default True)
            print labels along contours
        decimals : integer (default 0)
            number of decimals of labels along contours
        color : str or list of strings
            color of contour lines
        newfig : boolean (default True)
            create new figure
        figsize : tuple of 2 values (default is mpl default)
            size of figure
        legend : list or boolean (default True)
            add legend to figure
            if list of strings: use strings as names in legend
            
        """
        
        x1, x2, y1, y2 = win
        if np.isscalar(ngr):
            nx = ny = ngr
        else:
            nx, ny = ngr
        layers = np.atleast_1d(layers)
        xg = np.linspace(x1, x2, nx)
        yg = np.linspace(y1, y2, ny)
        h = self.headgrid(xg, yg, t, layers)
        if newfig:
            plt.figure(figsize=figsize)
        # color
        if color is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif type(color) is str:
            c = len(layers) * [color]
        elif type(color) is list:
            c = color
        if len(c) < len(layers):
            n = np.ceil(self.aq.naq / len(c))
            c = n * c 
        # contour
        cscollectionlist = []
        for i in range(len(layers)):
            cs = plt.contour(xg, yg, h[i, 0], levels, colors=c[i], 
                             linestyles='-')
            cscollectionlist.append(cs.collections[0])
            if labels:
                fmt = '%1.' + str(decimals) + 'f'
                plt.clabel(cs, fmt=fmt)
        if type(legend) is list:
            plt.legend(cscollectionlist, legend)
        elif legend:
            legendlist = ['layer ' + str(i) for i in layers]
            plt.legend(cscollectionlist, legendlist)
        plt.axis('scaled')
        if layout:
            self.plot(win=[x1, x2, y1, y2], newfig=False)