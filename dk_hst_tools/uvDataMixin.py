import logging

import astropy.units as u
from astropy.coordinates import SkyCoord, concatenate
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from pymccorrelation import pymccorrelation



class UVSpectraMixin(object):
    """
    Mixin class with convenience functionality for HST COS Data

    """

    def get_SkyCoords(self):
        """
        Get SkyCoords for each entry in table

        Parameters
        ----------
        
        """

        return concatenate(([self.coords_dict[source_name] for source_name in self["SOURCE"]]))




    
    def get_angular_impact_parameter(self, target_coordinate, coords):
        """
        Calculate impact parameter (projected angular distance from target)

        Parameters
        ----------
        target_coordinate: `astropy.coordinates.SkyCoord`
            Target galaxy coordinate
        coords: `astropy.coordinates.SkyCoord`
            list of coordinates to calculate impact parameter for


        """
        return target_coordinate.separation(coords)

    def get_LMC_impact_parameter(self, coords, distance = 50*u.kpc):
        """
        Calculate impact parameter to LMC assuming some distances

        Parameters
        ----------
        coords: `astropy.coordinates.SkyCoord`
            list of coordinates to calculate impact parameter for
        distance: `u.Quantity, number`, optional, must be keyword
            assumed distance to LMC
        """

        if not hasattr(distance, "unit"):
            distance *= u.kpc
            logging.warning("No units provided for distance, assuming kpc!")

        return np.tan(self.LMC_coords.separation(coords)) * distance

    def get_LMC_mccorrelation(self, 
                              ion, 
                              coeff = "kendallt", 
                              Nboot = 1000, 
                              as_dict = True,
                              **kwargs):
        """
        Calculate rank correlation coefficients using bootstrapping and censoring

        Parameters
        ----------
        ion:  `str`
            Name of ion
        coeff: `str`, optional, must be keyword
            rank coefficient to calculate
        Nboot: `number`, optional, must be keyword
            number of bootstrap resamples to draw 
        as_dict: `bool`, optional ,must be keyword
            if True, returns as labeled dictionary of entries
        **kwargs:
            keywords passed to pymccorrelation
        """

        x = self["LMC_B"]
        y = self["N_{}".format(ion)]
        ylim = np.zeros_like(x, dtype = int)
        ylim[self["N_{}_LOWERLIMIT".format(ion)]] = -1
        ylim[self["N_{}_UPPERLIMIT".format(ion)]] = 1

        xlim = np.zeros_like(x, dtype = int)

        res = pymccorrelation(x, y, xlim = xlim, ylim = ylim, 
            coeff = coeff, Nboot = Nboot, **kwargs)

        if as_dict:
            out = {}
            if "return_dist" in kwargs:
                if kwargs["return_dist"]:
                    out["coeff_percentiles"] = res[0]
                    out["p-value_percentiles"] = res[1]
                    out["coeff_dist"] = res[2]
                    out["p-value_dist"] = res[3]
                else:
                    out["coeff_percentiles"] = res[0]
                    out["p-value_percentiles"] = res[1]
            else:
                out["coeff_percentiles"] = res[0]
                out["p-value_percentiles"] = res[1]

            return out
        else:
            return res








    def plot_NvB_LMC(self, ion, 
                     ax = None, 
                     log = False, 
                     add_labels = False, 
                     colorbar = False,
                     length_factor = 0.25,
                     head_length_factor = 0.075,
                     limit_kwargs = {},
                     colorbar_kwargs = {}, 
                     **kwargs):
        """
        Make a plot of column density vs. impact parameter to LMC, including accounting for upper limits

        Parameters
        ----------

        ion: `str`
            name of ion to plot column density for
        ax: `matplotlib.pyplot.axis', optional, must be keyword
            if provided, plots on provided axis, else makes new figure and axis
        log: `bool`, optional, must be keyword
            if True, plots log column densities
        add_labels: `bool`, optional, must be keyword
            if True, adds axes labels
        colorbar: `bool`, optional, must be keyword
            if True, add colorbar
        length_factor:, `number`, optional, must be keyword
            factor to set length of limit arrows
        head_length_factor: `number`, optional must be keyword
            if not log, then used to set length scaling of arrow head
        limit_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to pass to plotting limit markers
        colorbar_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to pass to plotting of colorbar
        **kwargs: keywords passed to ax.scatter
        
        Returns
        -------
        ax: `matplotlib.pyplot.axis`
        """

        # Make sure ion exists in data
        if "N_{}".format(ion) not in self.keys():
            raise ValueError("Specified ion is not in this data!")

        if ax is None:
            fig, ax = plt.subplots()

        # Check limit kwargs
        if ("color" in kwargs) & ("color" not in limit_kwargs):
            limit_kwargs["color"] = kwargs["color"]
        if ("lw" not in limit_kwargs):
            limit_kwargs["lw"] = 2
        if ("alpha" in kwargs) & ("alpha" not in limit_kwargs):
            limit_kwargs["alpha"] = kwargs["alpha"]

        if ("zorder" not in kwargs):
            kwargs["zorder"] = 1
        if ("zorder" in kwargs) & ("zorder" not in limit_kwargs):
            limit_kwargs["zorder"] = kwargs["zorder"] - 1


        if "label" not in kwargs:
            kwargs["label"] = ion

        if ("c" in kwargs) & ("norm" not in kwargs):
            if "vmin" not in kwargs:
                vmin = np.nanmin(kwargs["c"])
            else:
                vmin = kwargs.pop("vmin")
            if "vmax" not in kwargs:
                vmax = np.nanmax(kwargs["c"])
            else:
                vmax = kwargs.pop("vmax")

            norm = Normalize(vmin = vmin, vmax = vmax)
        elif ("c" in kwargs) & ("norm" in kwargs):
            norm = kwargs.pop("norm")

        if ("c" in kwargs):
            if len(kwargs["c"].shape) != 2:
                c = kwargs.pop("c")
                if "cmap" not in kwargs:
                    cmap = "viridis"
                else:
                    cmap = kwargs.pop("cmap")
                cmapper = cm.ScalarMappable(cmap = cmap, norm = norm)
                kwargs["c"] = cmapper.to_rgba(c)

            need_colorhelp = True

        else:
            need_colorhelp = False

        # Get x and y values to plot
        x = self["LMC_B"].copy()
        xlabel = "LMC Impact Parameter (kpc)"
        y = self["N_{}".format(ion)].copy()
        ylabel = r"Column Density $(cm^{-2})$"
        if log:
            y = np.log10(y)
            ylabel = r"$\log_{10}(N / cm^{-2})$"


        # Plot
        s = ax.scatter(x, y, **kwargs)

        color = s._facecolors[0]

        if add_labels:
            _ = ax.set_ylabel(ylabel, fontsize = 12)
            _ = ax.set_xlabel(xlabel, fontsize = 12)

        if not log:
            ax.set_yscale("log")

        if colorbar:
            fig = plt.gcf()
            cb = fig.colorbar(cmapper, **colorbar_kwargs)

        # Handle upper limits
        # Upperlimits
        if "width" not in limit_kwargs:
            limit_kwargs["width"] = 0.1
        if "head_width" not in limit_kwargs:
            limit_kwargs["head_width"] = 1
        if "overhang" not in limit_kwargs:
            limit_kwargs["overhang"] = .5

        try:
            default_head_length = limit_kwargs.pop("head_length")
        except KeyError:
            default_head_length = None




        for ell,row in enumerate(self[self["N_{}_UPPERLIMIT".format(ion)]]):
            x = row["LMC_B"]
            y = row["N_{}".format(ion)].copy()
            dy = length_factor*y
            if log:
                y = np.log10(y)
                dy = length_factor/2.
                if default_head_length is None:
                    head_length = 0.05
                else:
                    head_length = default_head_length
            else:
                if default_head_length is None:
                    head_length = head_length_factor*y
                else:
                    head_length = default_head_length

            if need_colorhelp:
                c_up = kwargs["c"][self["N_{}_UPPERLIMIT".format(ion)],:]
                color = c_up[ell,:]

            elif "color" in limit_kwargs:
                color = limit_kwargs.pop("color")

            

            ax.arrow(x, y, 0, -dy, head_length = head_length, color = color, **limit_kwargs)

        for ell,row in enumerate(self[self["N_{}_LOWERLIMIT".format(ion)]]):
            x = row["LMC_B"]
            y = row["N_{}".format(ion)].copy()
            dy = length_factor*y
            if log:
                y = np.log10(y)
                dy = length_factor/2.
                if default_head_length is None:
                    head_length = 0.05
                else:
                    head_length = default_head_length
            else:
                if default_head_length is None:
                    head_length = head_length_factor*y
                else:
                    head_length = default_head_length

            if need_colorhelp:
                c_up = kwargs["c"][self["N_{}_LOWERLIMIT".format(ion)],:]
                color = c_up[ell,:]

            elif "color" in limit_kwargs:
                color = limit_kwargs.pop("color")

            

            ax.arrow(x, y, 0, dy, head_length = head_length, color = color, **limit_kwargs)

        return ax

























