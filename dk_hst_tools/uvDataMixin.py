import logging

import astropy.units as u
from astropy.coordinates import SkyCoord, concatenate
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from pymccorrelation import pymccorrelation

from pykrige.uk import UniversalKriging



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





    def get_UK(self, ion, frame = None, coord_names = ["default"], mask_limits = False, **kwargs):
        """
        Setup Universal Krigging on specified ion column densities

        Parameters
        ----------

        ion:`str`
            name of ion
        frame: `str`, `astropy.coordinates type`
            coordinate frame to use
        coord_names: `list of strings`, optional must be keyword
            used if using a custom frame not using l,b or ra,dec names
        mask_limits: `bool`, optional, must be keyword
            if True, masks out upper and lower limit values for kriging
        **kwargs: passed to pykrige.uk.UniversalKriging
        
        Returns
        -------
        uk: `pykrige.uk.UniversalKriging`
        """

        if frame is None:
            frame = 'galactic'


        try:
            coords = self.get_SkyCoords().transform_to(frame)
        except ValueError:
            logging.warning("frame not recognized, default to 'galactic'")
            coords = self.get_SkyCoords().transform_to("galactic")

        if coord_names[0].lower() == "default":
            try:
                x = coords.l.value
                y = coords.b.value
            except AttributeError:
                x = coords.ra.to(u.deg).value
                y = coords.dec.to(u.deg).value

        else:
            try:
                exec("x = coords.{}.value".format(coord_names[0]))
                exec("y = coords.{}.value".format(coord_names[1]))
            except AttributeError:
                raise AttributeError("coord_names provided not found!")

        z = self["N_{}".format(ion)]

        mask = np.isnan(z)
        mask |= np.isinf(z)
        if mask_limits:
            mask |= self["N_{}_UPPERLIMIT".format(ion)]
            mask |= self["N_{}_LOWERLIMIT".format(ion)]

        # keyword defaults
        if "variogram_model" not in kwargs:
            kwargs["variogram_model"] = "exponential"
        if "drift_terms" not in kwargs:
            kwargs["drift_terms"] = ["regional_linear"]

        return UniversalKriging(
            x[~mask],
            y[~mask],
            z[~mask],
            **kwargs
            )


    def UK_predict(self, 
                   ion = None, 
                   UK = None, 
                   frame = None, 
                   coord_names = ["default"], 
                   extent = None,
                   gridx = None,
                   gridy = None,
                   dxy = None,
                   mask_limits = False,
                   **kwargs):
        """
        execute Universal Krigging for privded extent or grid

        Parameters
        ----------
        ion:`str`
            name of ion
        UK: `pykrige.uk.UniversalKrigging'
            pre-computed UniversalKrigging
        frame: `str`, `astropy.coordinates type`
            coordinate frame to use
        coord_names: `list of strings`, optional must be keyword
            used if using a custom frame not using l,b or ra,dec names
        extent: `list-like`
            range of coordinates to grid across [xmin,xmax,ymin,ymax]
        gridx: `list-like`
            x axis grid values
        gridy: `list-like`
            y axis grid values
        dxy:   `number`
            step size if only extent is provided

            
        Return:
        z: `np.array`
            predicted values on grid
        ss: `np.array`
            variance at grid points
        """

        if frame is None:
            frame = "galactic"

        if dxy is None:
            dxy = 1.

        if UK is None:
            UK = self.get_UK(ion, frame = frame, coord_names = coord_names, 
                             mask_limits = mask_limits, **kwargs)

        if extent is None:
            return UK.execute("grid", gridx, gridy)

        else:
            gridx = np.arange(extent[0], extent[1], dxy)
            gridy = np.arange(extent[2], extent[3], dxy)

            return UK.execute("grid", gridx, gridy)




    def plot_column_map(self, 
                        ion,
                        ax = None, 
                        log = False, 
                        add_labels = True,
                        colorbar = False,
                        add_LMC_marker = True,
                        add_LMC_label = True, 
                        krig = False,
                        UK = None,
                        mask_limits_for_krig = True,
                        frame = None,
                        coord_names = ["default"],
                        extent = None,
                        gridx = None, 
                        gridy = None,
                        dxy = None,
                        wrap_at = None,
                        return_krig_ss = False,
                        LMC_kwargs = {},
                        LMC_label_kwargs = {},
                        colorbar_kwargs = {},
                        krig_kwargs = {},
                        im_kwargs = {},
                        **kwargs):
        """
        Make a scatter plot map color coded by column density
        Warning: does not account for upper or lower limits currently for krigging

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
        add_LMC_marker: `bool`, optional, must be keyword
            if True, plots marker for LMC location
        add_LMC_label: `bool`, optional, must be keyword
            if True, adds label for LMC location
        krig: `bool`, optional, must be keyword
            if True, interpolates with krigging
        return_krig_ss: `bool`, optional, must be keyword
            if True, also returns krig_ss
        UK: `pykrige.uk.UniversalKriging`
            precomputed kriging model
        mask_limits_for_krig: `bool`, optional, must be keyword
            if True, masks out upper and lower limit values for kriging
        frame: `str`, optional, must be keyword
            frame to use for kriging
        coord_names: `list of strings`, optional must be keyword
            used if using a custom frame not using l,b or ra,dec names
        wrap_at: `str`, optional, must be keyword
            angle to wrap longitude at
        LMC_kwargs: `dict`
            keywords for adding LMC marker
        LMC_label_kwargs: `dict`
            keywords for LMC label text
        colorbar_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to pass to plotting of colorbar
        krig_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to pass to kriging method
        im_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to imshow for krig
        extent: `list-like`
            range of coordinates to grid across [xmin,xmax,ymin,ymax]
        gridx: `list-like`
            x axis grid values
        gridy: `list-like`
            y axis grid values
        dxy:   `number`
            step size if only extent is provided
        **kwargs: keywords passed to ax.scatter
        
        """

        # Make sure ion exists in data
        if "N_{}".format(ion) not in self.keys():
            raise ValueError("Specified ion is not in this data!")

        if frame is None:
            frame = "galactic"

        if wrap_at is None:
            wrap_at = "360d"

        if ax is None:
            fig, ax = plt.subplots()

        # get coordinates to plot:

        try:
            coords = self.get_SkyCoords().transform_to(frame)
        except ValueError:
            logging.warning("frame not recognized, default to 'galactic'")
            coords = self.get_SkyCoords().transform_to("galactic")

        if coord_names[0].lower() == "default":
            try:
                x = coords.l.wrap_at(wrap_at).value
                y = coords.b.value
            except AttributeError:
                x = coords.ra.to(u.deg).value
                y = coords.dec.to(u.deg).value

        else:
            try:
                exec("x = coords.{}.value".format(coord_names[0]))
                exec("y = coords.{}.value".format(coord_names[1]))
            except AttributeError:
                raise AttributeError("coord_names provided not found!")

        z = self["N_{}".format(ion)].copy()
        if log:
            z = np.log10(z)



        # Check for kwargs:
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.7
        if "norm" not in kwargs:
            if ("vmin" not in kwargs) & ("vmax" not in kwargs):
                kwargs["norm"] = Normalize(vmin = np.nanmin(z), vmax = np.nanmax(z))
            elif ("vmin" not in kwargs) & ("vmax" in kwargs):
                kwargs["norm"] = Normalize(vmin = np.nanmin(z), vmax = vmax)
            elif ("vmax" not in kwargs) & ("vmin" in kwargs):
                kwargs["norm"] = Normalize(vmin = vmin, vmax = np.nanmax(z))

        if "s" not in kwargs:
            kwargs["s"] = 100

        if ("edgecolor" not in kwargs) & (krig):
            kwargs["edgecolor"] = "w"

        s = ax.scatter(x, y, c = z, **kwargs)

        if colorbar:
            fig = plt.gcf()
            cb = fig.colorbar(s, **colorbar_kwargs)

        if add_labels:
            if frame == "galactic":
                xlabel = "Galactic Longitude (deg)"
                ylabel = "Galactic Latitude (deg)"
            elif frame == "icrs":
                xlabel = "RA (deg)"
                ylabel = "Dec (deg)"
            else:
                xlabel = coord_names[0]
                ylabel = coord_names[-1]

            _ = ax.set_xlabel(xlabel, fontsize = 12)
            _ = ax.set_ylabel(ylabel, fontsize = 12)

        if add_LMC_marker:
            #Check LMC_kwargs
            if "color" not in LMC_kwargs:
                LMC_kwargs["color"] = "k"
            if "s" not in LMC_kwargs:
                LMC_kwargs["s"] = 10000
            if "marker" not in LMC_kwargs:
                LMC_kwargs["marker"] = "*"
            if "alpha" not in LMC_kwargs:
                LMC_kwargs["alpha"] = 0.2
            if "zorder" not in LMC_kwargs:
                LMC_kwargs["zorder"] = -1

            if coord_names[0].lower() == "default":
                try:
                    lmc_x = self.LMC_coords.transform_to(frame).l.wrap_at(wrap_at).value
                    lmc_y = self.LMC_coords.transform_to(frame).b.value
                except AttributeError:
                    lmc_x = self.LMC_coords.transform_to(frame).ra.to(u.deg).value
                    lmc_y = self.LMC_coords.transform_to(frame).dec.to(u.deg).value
            else:
                exec("lmc_x = self.LMC_coords.transform_to(frame).{}.value".format(coord_names[0]))
                exec("lmc_y = self.LMC_coords.transform_to(frame).{}.value".format(coord_names[1]))



            lmc_s = ax.scatter(lmc_x, lmc_y, **LMC_kwargs)

        if add_LMC_label:
            # check kwargs
            if "ha" not in LMC_label_kwargs:
                LMC_label_kwargs["ha"] = "center"
            if "va" not in LMC_label_kwargs:
                LMC_label_kwargs["va"] = "center"
            if "fontweight" not in LMC_label_kwargs:
                LMC_label_kwargs["fontweight"] = "bold"
            if "fontsize" not in LMC_label_kwargs:
                LMC_label_kwargs["fontsize"] = 12
            if "color" not in LMC_label_kwargs:
                LMC_label_kwargs["color"] = "k"

            if coord_names[0].lower() == "default":
                try:
                    lmc_x = self.LMC_coords.transform_to(frame).l.wrap_at(wrap_at).value
                    lmc_y = self.LMC_coords.transform_to(frame).b.value
                except AttributeError:
                    lmc_x = self.LMC_coords.transform_to(frame).ra.to(u.deg).value
                    lmc_y = self.LMC_coords.transform_to(frame).dec.to(u.deg).value
            else:
                exec("lmc_x = self.LMC_coords.transform_to(frame).{}.value".format(coord_names[0]))
                exec("lmc_y = self.LMC_coords.transform_to(frame).{}.value".format(coord_names[1]))

            lmc_l = ax.text(lmc_x, lmc_y, "LMC", **LMC_label_kwargs)

        if (UK is not None) | (krig):

            if (extent is None) & (gridx is None) & (gridy is None):
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                extent = [xlim[0], xlim[1], ylim[0], ylim[1]]


            z_pred, ss_pred = self.UK_predict(ion = ion, 
                                     UK = UK, 
                                     frame = frame, 
                                     coord_names = coord_names,
                                     extent = extent,
                                     gridx = gridx,
                                     gridy = gridy,
                                     dxy = dxy, 
                                     mask_limits = mask_limits_for_krig, 
                                     **krig_kwargs)

            if log:
                z_pred = np.log10(z_pred)

            if extent is None:
                extent = [gridx[0], gridx[1], gridy[0], gridy[1]]



            # check kwargs
            if ("cmap" in kwargs) & ("cmap" not in im_kwargs):
                im_kwargs["cmap"] = kwargs["cmap"]
            if ("norm" not in im_kwargs):
                im_kwargs["norm"] = kwargs["norm"]
            if ("zorder" not in im_kwargs):
                im_kwargs["zorder"] = -2
       


            im = ax.imshow(z_pred, extent = extent, **im_kwargs)

            if return_krig_ss:
                return ax, ss_pred

        return ax
















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

























