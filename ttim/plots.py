from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class PlotTtim:
    def __init__(self, ml):
        self._ml = ml

    def topview(self, win=None, ax=None, figsize=None, layers=None):
        """Plot top-view.

        Parameters
        ----------
        win : list or tuple
            [x1, x2, y1, y2]
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        if layers is not None and isinstance(layers, int):
            layers = [layers]
        for e in self._ml.elementlist:
            if layers is None or np.isin(e.layers, layers):
                e.plot(ax=ax)
        if win is not None:
            ax.axis(win)
        return ax

    def xsection(
        self,
        xy: Optional[list[tuple[float]]] = None,
        labels=True,
        params=False,
        ax=None,
    ):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        if xy is not None:
            (x0, y0), (x1, y1) = xy
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            ax.set_xlim(0, r)
        else:
            r = 1.0

        if labels:
            lli = 1 if self._ml.aq.topboundary == "con" else 0
            aqi = 0
        else:
            lli = None
            aqi = None

        for i in range(self._ml.aq.nlayers):
            if self._ml.aq.ltype[i] == "l":
                ax.axhspan(
                    ymin=self._ml.aq.z[i + 1],
                    ymax=self._ml.aq.z[i],
                    color=[0.8, 0.8, 0.8],
                )
                if labels:
                    ax.text(
                        0.5 * r if not params else 0.25 * r,
                        np.mean(self._ml.aq.z[i : i + 2]),
                        f"leaky layer {lli}",
                        ha="center",
                        va="center",
                    )
                if params:
                    ax.text(
                        0.75 * r if labels else 0.5 * r,
                        np.mean(self._ml.aq.z[i : i + 2]),
                        (
                            f"$c$ = {self._ml.aq.c[lli]}, "
                            f"$S_s$ = {self._ml.aq.Sll[lli]:.2e}"
                        ),
                        ha="center",
                        va="center",
                    )
                if labels or params:
                    lli += 1

            if labels and self._ml.aq.ltype[i] == "a":
                ax.text(
                    0.5 * r if not params else 0.25 * r,
                    np.mean(self._ml.aq.z[i : i + 2]),
                    f"aquifer {aqi}",
                    ha="center",
                    va="center",
                )
            if params and self._ml.aq.ltype[i] == "a":
                if aqi == 0 and self._ml.aq.phreatictop:
                    paramtxt = (
                        f"$k_h$ = {self._ml.aq.kaq[aqi]}, "
                        f"$S$ = {self._ml.aq.Saq[aqi]}"
                    )
                else:
                    paramtxt = (
                        f"$k_h$ = {self._ml.aq.kaq[aqi]}, "
                        f"$S_s$ = {self._ml.aq.Saq[aqi]:.2e}"
                    )
                ax.text(
                    0.75 * r if labels else 0.5 * r,
                    np.mean(self._ml.aq.z[i : i + 2]),
                    paramtxt,
                    ha="center",
                    va="center",
                )
            if (labels or params) and self._ml.aq.ltype[i] == "a":
                aqi += 1

        for i in range(1, self._ml.aq.nlayers):
            if self._ml.aq.ltype[i] == "a" and self._ml.aq.ltype[i - 1] == "a":
                ax.axhspan(
                    ymin=self._ml.aq.z[i], ymax=self._ml.aq.z[i], color=[0.8, 0.8, 0.8]
                )

        ax.axhline(self._ml.aq.z[0], color="k", lw=0.75)
        ax.axhline(self._ml.aq.z[-1], color="k", lw=3.0)
        ax.set_ylabel("elevation")
        return ax

    def head_along_line(
        self,
        x1=0,
        x2=1,
        y1=0,
        y2=0,
        npoints=100,
        t=1.0,
        layers=0,
        sstart=0,
        color=None,
        lw=1,
        figsize=None,
        ax=None,
        legend=True,
        grid=True,
    ):
        layers = np.atleast_1d(layers)
        t = np.atleast_1d(t)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        s = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2) + sstart
        h = self._ml.headalongline(x, y, t, layers)
        nlayers, ntime, npoints = h.shape
        for i in range(nlayers):
            for j in range(ntime):
                lbl = f"head (t={t[j]}, layer={layers[i]})"
                if color is None:
                    ax.plot(s, h[i, j, :], lw=lw, label=lbl)
                else:
                    ax.plot(s, h[i, j, :], color, lw=lw, label=lbl)
        if legend:
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
        if grid:
            ax.grid(True)
        return ax

    def contour(
        self,
        win,
        ngr=20,
        t=1,
        layers=0,
        levels=20,
        layout=True,
        labels=True,
        decimals=1,
        color=None,
        ax=None,
        figsize=None,
        legend=True,
    ):
        """Contour plot.

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
        decimals : integer (default 1)
            number of decimals of labels along contours
        color : str or list of strings
            color of contour lines
        ax : matplotlib.Axes
            axes to plot on, default is None which creates a new figure
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
        h = self._ml.headgrid(xg, yg, t, layers)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", adjustable="box")
        # color
        if color is None:
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(color, str):
            c = len(layers) * [color]
        elif isinstance(color, list):
            c = color
        if len(c) < len(layers):
            n = np.ceil(self._ml.aq.naq / len(c))
            c = n * c

        # contour
        cslist = []
        cshandlelist = []
        for i in range(len(layers)):
            cs = ax.contour(
                xg, yg, h[i, 0], levels, colors=c[i], negative_linestyles="solid"
            )
            cslist.append(cs)
            handles, _ = cs.legend_elements()
            cshandlelist.append(handles[0])
            if labels:
                fmt = f"%1.{decimals}f"
                ax.clabel(cs, fmt=fmt)
        if isinstance(legend, list):
            ax.legend(cshandlelist, legend, loc=(0, 1), ncol=3, frameon=False)
        elif legend:
            legendlist = ["layer " + str(i) for i in layers]
            ax.legend(cshandlelist, legendlist, loc=(0, 1), ncol=3, frameon=False)

        if layout:
            self.topview(win=[x1, x2, y1, y2], ax=ax)
        return ax
