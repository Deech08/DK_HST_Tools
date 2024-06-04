import logging

import astropy.units as u
from astropy.coordinates import SkyCoord, concatenate
from astropy.constants import c as speed_of_light
from astropy.table import QTable, Table, hstack, vstack
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm

from pymccorrelation import pymccorrelation

from pykrige.uk import UniversalKriging

import os
import glob

from spectral_cube import SpectralCube


# from .sbiKit import sbiKit

from lmfit import Parameters

from VoigtFit.io.output import rebin_spectrum, rebin_bool_array




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

        return np.sin(self.LMC_coords.separation(coords)) * distance

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

        if ion.__class__ is str:
            z = self["N_{}".format(ion)].copy()
        else:
            z = np.zeros_like(self["N_{}".format(ion[0])])
            for ion_name in ion:
                z += self["N_{}".format(ion_name)].copy()

        mask = np.isnan(z)
        mask |= np.isinf(z)
        if mask_limits:
            if ion.__class__ is str:
                mask |= self["N_{}_UPPERLIMIT".format(ion)]
                mask |= self["N_{}_LOWERLIMIT".format(ion)]
            else:
                for ion_name in ion:
                    mask |= self["N_{}_UPPERLIMIT".format(ion_name)]
                    mask |= self["N_{}_LOWERLIMIT".format(ion_name)]

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
                        add_LMC_marker = False,
                        add_SMC_marker = False,
                        add_LMC_label = True, 
                        add_SMC_label = True, 
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
                        SMC_kwargs = {},
                        LMC_label_kwargs = {},
                        SMC_label_kwargs = {},
                        colorbar_kwargs = {},
                        krig_kwargs = {},
                        im_kwargs = {},
                        **kwargs):
        """
        Make a scatter plot map color coded by column density
        Warning: does not account for upper or lower limits currently for krigging

        Parameters
        ----------
        ion: `str`, `list-like`
            name of ion to plot column density for
            can also be list of ions to sum
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
        add_SMC_label: `bool`, optional, must be keyword
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
        SMC_kwargs: `dict`
            keywords for adding LMC marker
        LMC_label_kwargs: `dict`
            keywords for LMC label text
        SMC_label_kwargs: `dict`
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
        if ion.__class__ is str:
            z = self["N_{}".format(ion)].copy()
        else:
            z = np.zeros_like(self["N_{}".format(ion[0])])
            for ion_name in ion:
                z += self["N_{}".format(ion_name)].copy()

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
            if ("transform" in kwargs) & ("transform" not in LMC_kwargs):
                LMC_kwargs["transform"] = kwargs["transform"]

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
        if add_SMC_marker:
            #Check LMC_kwargs
            if "color" not in SMC_kwargs:
                SMC_kwargs["color"] = "k"
            if "s" not in SMC_kwargs:
                SMC_kwargs["s"] = 10000
            if "marker" not in SMC_kwargs:
                SMC_kwargs["marker"] = "*"
            if "alpha" not in SMC_kwargs:
                SMC_kwargs["alpha"] = 0.2
            if "zorder" not in SMC_kwargs:
                SMC_kwargs["zorder"] = -1
            if ("transform" in kwargs) & ("transform" not in SMC_kwargs):
                SMC_kwargs["transform"] = kwargs["transform"]

            if coord_names[0].lower() == "default":
                try:
                    smc_x = self.SMC_coords.transform_to(frame).l.wrap_at(wrap_at).value
                    smc_y = self.SMC_coords.transform_to(frame).b.value
                except AttributeError:
                    smc_x = self.SMC_coords.transform_to(frame).ra.to(u.deg).value
                    smc_y = self.SMC_coords.transform_to(frame).dec.to(u.deg).value
            else:
                exec("smc_x = self.SMC_coords.transform_to(frame).{}.value".format(coord_names[0]))
                exec("smc_y = self.SMC_coords.transform_to(frame).{}.value".format(coord_names[1]))



            smc_s = ax.scatter(smc_x, smc_y, **SMC_kwargs)

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
            if ("transform" in kwargs) & ("transform" not in LMC_label_kwargs):
                LMC_label_kwargs["transform"] = kwargs["transform"]

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

        if add_SMC_label:
            # check kwargs
            if "ha" not in SMC_label_kwargs:
                SMC_label_kwargs["ha"] = "center"
            if "va" not in SMC_label_kwargs:
                SMC_label_kwargs["va"] = "center"
            if "fontweight" not in SMC_label_kwargs:
                SMC_label_kwargs["fontweight"] = "bold"
            if "fontsize" not in SMC_label_kwargs:
                SMC_label_kwargs["fontsize"] = 12
            if "color" not in SMC_label_kwargs:
                SMC_label_kwargs["color"] = "k"
            if ("transform" in kwargs) & ("transform" not in SMC_label_kwargs):
                SMC_label_kwargs["transform"] = kwargs["transform"]

            if coord_names[0].lower() == "default":
                try:
                    smc_x = self.SMC_coords.transform_to(frame).l.wrap_at(wrap_at).value
                    smc_y = self.SMC_coords.transform_to(frame).b.value
                except AttributeError:
                    smc_x = self.SMC_coords.transform_to(frame).ra.to(u.deg).value
                    smc_y = self.SMC_coords.transform_to(frame).dec.to(u.deg).value
            else:
                exec("smc_x = self.SMC_coords.transform_to(frame).{}.value".format(coord_names[0]))
                exec("smc_y = self.SMC_coords.transform_to(frame).{}.value".format(coord_names[1]))

            smc_l = ax.text(smc_x, smc_y, "SMC", **SMC_label_kwargs)

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


    def plot_impact_parameter_contour(self, 
                                      ax = None,
                                      levels = None, 
                                      keep_axlims = True,
                                      add_labels = True,
                                      **kwargs):
        """
        Plot contours of impact parameter on map

        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional, must keyword
            axis to plot onto, or creates new one
        levels: `list-like`, optional, must be keyword
            contour levels to plot
        keep_axlims: `bool`, optional, must be keyword
            if True, keeps existing xlim and ylim values
        add_labels: `bool`, optional, must be keyword
            if True, labels contour levels



        """

        # Check for ax
        if ax is None:
            fig, ax = plt.subplots()

        # Check for wcs
        if hasattr(ax, "wcs"):
            if "transform" not in kwargs:
                kwargs["transform"] = ax.get_transform("world")

        if levels is None:
            levels = [5, 10, 20, 30, 40, 50]

        # Get coordinates:
        if keep_axlims:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        gridx = np.arange(240,340,1.)
        gridy = np.arange(-85,15,1.)

        xx, yy = np.meshgrid(gridx, gridy, indexing = "ij")

        coords = SkyCoord(l = xx*u.deg, b = yy*u.deg, frame = "galactic")
        z = self.get_LMC_impact_parameter(coords)

        # plot contours

        ct = ax.contour(xx, yy, z, levels = levels, **kwargs)

        if keep_axlims:
            xlim = ax.set_xlim(xlim)
            ylim = ax.set_ylim(ylim)

        if add_labels:
            fmt = {}
            for l in ct.levels:
                fmt[l] = "{} kpc".format(l)
            ax.clabel(ct, ct.levels, fmt = fmt)

        return ax 








    def plot_LMC_HI_map(self, 
                        cube = None,
                        ax = None, 
                        region_str = None, 
                        clip_mask = True, 
                        clip_value = None, 
                        use_wcs = True,
                        fit_to_ax = False, 
                        vel_range = None,
                        colorbar = False,
                        colorbar_kwargs = {},
                        frame = None,
                        order = None,
                        column_density = True,
                        **kwargs):
        """
        Plots HI Map

        Parameters
        ----------
        cube: `spectral_cube.spectral_cube.SpectralCube`. optional ,must be keyword
            data cube to plot map from. if not provided, loads default map for DK
        ax: `matplotlib.pyplot.axes`, optional, must keyword
            axis to plot onto, or creates new one
        region_str: `str`, optional, must be keyword
            DS9 region string to cut cube to
        clip_mask: `bool`, optional, must be keyword
            if True, clips data at clip_value
        clip_value: `number, astropy.units.quantity`, optional, must be keyword
            value to clip intensity at, defaults to 10^18 cm^-2
        use_wcs: `bool`, optional, must be keyword
            determins whether to use WCS axes or not
        fit_to_ax: `bool`, optional must be keyword
            if True, clips cube to fit size of existing axis
        vel_range: `list-like`, optional, must be keyword:
            velocity range to integrate cube over, assumes km/s units if not provided
        colorbar: `bool`, optional, must be keyword
            if True, adds colorbar
        colorbar_kwargs: `dict`, optional, must be keyword
            dictionary of keywords to pass to plotting of colorbar
        frame: `str`, optional ,must be keyword
            Coordiante frame to use
        order: `int`, optional, must be keyword
            order of moment to plot, default of 0
        column_density: `bool`, optional, must be keyword
            if True, converts order 0 to a column density
        kwargs: passed to ax.imshow

        Returns
        -------
        ax: `matplotlib.pyplot.Axes'
        """

        # Get data cube
        if cube is None:
            #load custom_cut_cube
            cube = os.path.join(self.path, "../HI/hi4pi_LMC_cut.fits")

        if cube.__class__ is str:
            cube = SpectralCube.read(cube)


        # Check for region_str:
        if region_str is not None:
            cube = cube.subcube_from_ds9region(region_str)
        elif (fit_to_ax is True) & (ax is not None):
            # Frame limitation:
            if frame is None:
                frame = "galactic"
            elif frame != "galactic":
                raise NotImplementedError("Sorry, only implemented for Galactic Coordiantes so far!")
            # Currently only works with Galactic Coordiantes
            if hasattr(ax, "wcs"):
                xlim_raw = ax.get_xlim()
                ylim_raw = ax.get_ylim()
                c_lower_left, c_lower_right, c_upper_left, c_upper_right = ax.wcs.pixel_to_world(
                                                [xlim_raw[0], 
                                                 xlim_raw[1], 
                                                 xlim_raw[0], 
                                                 xlim_raw[1]], 
                                                [ylim_raw[0], 
                                                 ylim_raw[0], 
                                                 ylim_raw[1], 
                                                 ylim_raw[1]]
                    )
                center = ax.wcs.pixel_to_world(np.median(xlim_raw), np.median(ylim_raw))
                dx = np.abs(c_lower_right.l.value - c_lower_left.l.value)
                dy = np.abs(c_lower_left.b.value - c_upper_left.b.value)

                region_str = "galactic; box({}, {}, {}, {})".format(center.l.value, 
                                                                    center.b.value, 
                                                                    dx, 
                                                                    dy)

                cube = cube.subcube_from_ds9region(region_str)

            else:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                center_x, center_y =  np.median(xlim), np.median(ylim)
                dx = np.abs(np.diff(xlim))[0]
                dy = np.abs(np.diff(ylim))[0]

                if "extent" not in kwargs:
                    kwargs["extent"] = [xlim[0], xlim[1], ylim[0], ylim[1]]

                region_str = "galactic; box({}, {}, {}, {})".format(center_x, 
                                                                    center_y, 
                                                                    dx, 
                                                                    dy)


                cube = cube.subcube_from_ds9region(region_str)

        # set_velrange
        if vel_range is None:
            vel_range = [150,350] * u.km/u.s
        elif not hasattr(vel_range, "unit"):
            vel_range *= u.km/u.s
            logging.warning("No units for vel_range provided, assuming u.km/u.s")

        # Apply spectral cut
        cube = cube.spectral_slab(vel_range[0], vel_range[1])


        # Compute moment

        col_den_factor = 1.823*10**18 * u.cm**-2 / (u.K * u.km/u.s)

        if clip_value is None:
            clip_value = 1e18* u.cm**-2
        elif not hasattr(clip_value, "unit"):
            clip_value *= u.cm**-2
            logging.warning("No units for clip_value, provided, assuming u.cm^-2")

        if order is None:
            order = 0

        moment = cube.moment(order = order)
        if order != 0:
            zero_moment = cube.moment(order = 0).to(u.K * u.km/u.s)
            zero_moment *= col_den_factor

        if order == 0:
            moment = moment.to(u.K * u.km/u.s) * col_den_factor
            if "cmap" not in kwargs:
                kwargs["cmap"] = "Greys"
            if "norm" not in kwargs:
                if column_density:
                    kwargs["norm"] = LogNorm(vmin = clip_value.to(u.cm**-2).value, 
                                             vmax = 1e22)
                else:
                    kwargs["norm"] = LogNorm(vmin = 3, vmax = 3000)
            if column_density:
                label = r"$N_{HI} (cm^{-2})$"
            else:
                label = "HI Intensity (K km / s)"
        elif order == 1:
            moment = moment.to(u.km/u.s)
            if "cmap" not in kwargs:
                kwargs["cmap"] = "RdBu_r"
            if "norm" not in kwargs:
                kwargs["norm"] = Normalize(vmin = vel_range[0].value, vmax = vel_range[1].value)
            label = "Mean Velocity (km / s)"
        elif order == 2:
            moment = moment.to(u.km**2/u.s**2)
            if "norm" not in kwargs:
                kwargs["norm"] = Normalize(vmin = 0, vmax = 50**2)
            label = r"Variance ((km / s)$^{2}$)"


        # apply clipping
        if clip_mask:

            if order != 0:
                masked_moment = np.ma.masked_array(moment.value, mask = zero_moment < clip_value)
            else:
                masked_moment = np.ma.masked_array(moment.value, mask = moment < clip_value)

        else:
            masked_moment = np.ma.masked_array(moment.value, mask = np.isnan(moment))


        # Plot map

        # Check for ax
        if ax is None:
            if use_wcs:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection = moment.wcs)

            else:
                fig, ax = plt.subplots()




        if use_wcs:
            if "transform" not in kwargs:
                kwargs["transform"] = ax.get_transform(moment.wcs)

            
            im = ax.imshow(masked_moment, **kwargs)
        else:
            if "origin" not in kwargs:
                kwargs["origin"] = "lower"

            if "extent" not in kwargs:
                # set extent from map lims
                raise NotImplementedError("Extent is missing, not yet implemented to auto set for image!")

            im = ax.imshow(masked_moment, **kwargs)

        if colorbar:
            fig = plt.gcf()
            if "label" not in colorbar_kwargs:
                colorbar_kwargs["label"] = label
            cb = fig.colorbar(im, **colorbar_kwargs)

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

        ion: `str`, 'listlike'
            name of ion to plot column density for
            can also be list of ions to sum column densities 
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

        if ion.__class__ is str:
            y = self["N_{}".format(ion)].copy()
        else:
            y = np.zeros_like(self["N_{}".format(ion[0])])
            for ion_name in ion:
                y += self["N_{}".format(ion_name)].copy()
        
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







    # # SBI Related functions
    # # pytorch
    # import torch
    # # sbi
    # import sbi.utils as utils
    # from sbi.inference.base import infer

    def exponential_halo_simulator(self, b, n_0, h_r, distance = 50, ):
        """
        exponential halo model at constant metallicity

        Parameters
        ----------
        b: `torch.tensor`, `list-like`, 
            impact parameter
        n_0: `number` - params[0]
            central density
        h_r: `number` - params[0]
            scale radius
        """

        # pytorch
        import torch
        # sbi
        import sbi.utils as utils
        from sbi.inference.base import infer

        # n_0, h_r = params

        # angular impact parameter
        alpha = torch.arctan(b/distance)
        def r(theta, alpha = alpha, D = distance, b = b):
            return D * torch.sin(alpha) / (torch.sin(np.pi - torch.abs(theta[:,None]) - torch.arcsin(D/b * torch.sin(alpha))))

        def dens(r):
            return n_0 * torch.exp(-1/2 * (r / h_r)**2)

        theta_grid = torch.linspace(-np.pi/2, np.pi/2, 1000)
        r_grid = r(theta_grid)
        dl_grid = torch.stack([r_grid[ell]**2 + r_grid[ell+2]**2 - 2 * r_grid[ell] * r_grid[ell+2] * 
            torch.cos(theta_grid[ell+2] - theta_grid[ell]) for ell in range(len(theta_grid)-2)])

        # convert dl_grid to cm
        dl_grid *= 3.08e18
    

        rcen_grid = r_grid[1:-1]

        dens_grid = dens(rcen_grid)
        col_dens_grid = dens_grid * dl_grid
        return torch.sum(col_dens_grid, axis = 0)

    def setup_SBI(self, ion, mask_limits = True, from_file = None):
        """
        setup SBI sbiKit

        Parameters
        ----------
        UVSpectra: 
            local UV spectra class
        ion: `str`, `list-like`:
            ion or list of ions to sum
        mask_limits: `bool`, optional, must be keyword
            if True, masks upper and lower limits
        from_file: `str`, optional, must be keyword
            if provided, loads sbiKit from filename provided
        """
        from .sbiKit import sbiKit

        return sbiKit(self, ion, mask_limits = mask_limits, from_file = from_file)

    def get_profile(self, ion_wav, pars, wl_arr = None, vel_arr = None, 
                    resolution = None, redshift = None, **kwargs):
        """
        Get absorption line profile for specified ion_wavelength

        Parameters
        ----------
        ion_wav: `str`
            ion_wavelength to get profile for
        pars: `dict`
            parameters for line profile
            must include "v", "b", and "logN"
        resolution: `number`, optional, must be keyword
            instrument resolution for convolution in km/s
            defualt to 20 km/s
        redshift: `number`, optional, must be keyword
            system redshift, default to 0
        wl_arr: `list-like`, optional, must be keyword
            array of wavelengths to compute spectrum to
        vel_arr: `list_like`, optional, must be keyword
            array of velocities to compute spectrum to
        """
        from VoigtFit.container.lines import show_transitions
        from VoigtFit.funcs.voigt import Voigt, convolve_numba
        from scipy.signal import fftconvolve
        from scipy.signal.windows import gaussian

        # find the right transition
        atomic_data = show_transitions(ion = ion_wav.split("_")[0])
        names = [at[0] for at in atomic_data]
        match = np.array(names) == ion_wav

        _, _, l0, f, gam, _ = np.array(atomic_data)[match][0]

        # check for resolution
        if resolution == None:
            resolution = 20. #km/s for COS

        # check redshift
        if redshift == None:
            redshift = 0.

        v = pars["v"]
        b = pars["b"]
        logN = pars["logN"]

        if ((type(v) == float) | (type(v) == np.float64)):
            v = list([v])
            b = list([b])
            logN = list([logN])

        l_center = l0 * (redshift + 1.)

        # find wl_line
        if vel_arr == None:
            if wl_arr == None:
                # use default wavelength range of +-500 km/s
                wl_arr = np.arange(l_center - .01*250, l_center +0.01*250, 0.01)

            vel_arr = (wl_arr - l_center)/l_center*(speed_of_light.to(u.km/u.s).value)

        elif wl_arr == None:
            wl_arr = (vel_arr / (speed_of_light.to(u.km/u.s).value) * l_center) + l_center

        tau = np.zeros_like(vel_arr)

        for (vv, bb, NN) in zip(v, b, logN):
            tau += Voigt(wl_arr, l0, f, 10**NN, 1.e5*bb, gam, z = vv/(speed_of_light.to(u.km/u.s).value))


        # Compute profile
        profile_int = np.exp(-tau)

        # convolve with instrument profile
        if isinstance(resolution, float):
            pxs = np.diff(wl_arr)[0] / wl_arr[0] * (speed_of_light.to(u.km/u.s).value)
            sigma_instrumental = resolution / 2.35482 / pxs
            LSF = gaussian(len(wl_arr) // 2, sigma_instrumental)
            LSF = LSF/LSF.sum()
            profile = fftconvolve(profile_int, LSF, 'same')
        else:
            profile = voigt.convolve_numba(profile_int, resolution)



        out = {
            "wl":wl_arr,
            "vel":vel_arr,
            "spec":profile
        }

        return out







    def plot_source_spectra(self, source_name,
                            figsize = None,
                            hspace = None,
                            wspace = None,
                            include_fits = True, 
                            include_components = True,
                            lines = None, 
                            n_cols = 2,
                            vel_range = None, ):
        """
        Summary Plot of all fits
        """

        # get relevent data

        fit_results = self.voigtfit[source_name]
        fit_flags = self.voigtfit_flags[source_name]

        # create figure

        if figsize == None:
            figsize = (8.5,11)
        fig = plt.figure(figsize = figsize)

        # check on ions to include
        if lines == None:
            lines = [
                "OI_1302",
                "CII_1334",
                "CIIa_1335.71",
                "SiII_1190", 
                "SiII_1193", 
                "SiII_1260",
                "SiII_1304",
                "SiII_1526",
                "SiIII_1206", 
                "FeII_1608",
                "FeII_1144",
                "AlII_1670",
                "CIV_1548",
                "CIV_1550",
                "SiIV_1393", 
                "SiIV_1402",
            ]

        n_lines = len(lines)

        n_rows = int(np.ceil(n_lines / n_cols))

        if hspace == None:
            hspace = 0.3
        if wspace == None:
            wspace = 0.2

        # create grid
        gs = plt.GridSpec(n_rows,n_cols,hspace = hspace, wspace = wspace)


        if vel_range == None:
            vel_range = [-500.,500.]

        print(vel_range)


        axs = {}
        current_col = 0
        current_row = 0
        for tag in lines:
            if current_row < n_rows:
                axs[tag] = fig.add_subplot(gs[current_row,current_col])
                current_row += 1
            else:
                current_row = 0
                current_col += 1
                axs[tag] = fig.add_subplot(gs[current_row,current_col])
                current_row += 1
            axs[tag].tick_params(axis='both', which='both', labelsize=8)
            axs[tag].set_xlim(vel_range)
            if tag == "CIIa_1335.71":
                axs[tag].set_ylabel(r"CII* $\lambda$1335", fontsize = 10)
            else:
                axs[tag].set_ylabel(r"{0} $\lambda${1}".format(*tag.split("_")), fontsize = 10)
                
                
        # Step through each region and plot fits
        def plot_fits(self, ions = "LOW"):
            n_regions = len(fit_results[ions].dataset.regions)
            repeat = False
            SiIV_0 = False
            SiIV_1 = False
            for ell in range(n_regions):
                # get tag
                lines = fit_results[ions].dataset.regions[ell].lines
                for sub_region_ind,line in enumerate(lines):
                    tag = line.tag
                    if (tag == "SiIV_1393") & (SiIV_0 == False):
                        SiIV_0 = True
                        tag = "SiIV_1393__0"
                    elif (tag == "SiIV_1402") & (SiIV_1 == False):
                        SiIV_1 = True
                        tag = "SiIV_1402__0"
                    if (tag == "SiIV_1393") & (SiIV_0 == True):
                        SiIV_0 = False
                        tag = "SiIV_1393__1"
                    elif (tag == "SiIV_1402") & (SiIV_1 == True):
                        SiIV_1 = False
                        tag = "SiIV_1402__1"
                        
                    try:    
                        _ = fit_results[ions].plot_region_fit(ell, sub_region_ind = sub_region_ind, 
                                                          vel_range = vel_range,
                                                          ax = axs[tag.split("__")[0]], 
                                                          labelx = False,
                                                          labely = False, 
                                                          ylabel_as_ion = True,
                                                          lw = 1, 
                                                          alpha = 0.6,
                                                          fit_kwargs = {"lw":1, "alpha":0.7},
                                                          comp_scale = .2, 
                                                          plot_indiv_comps = True)
                    except KeyError:
                        # skip
                        pass
                        
        
        plot_fits(self, ions = "LOW")
        plot_fits(self, ions = "HIGH")
        
        for tag in lines:
            ylim = axs[tag].get_ylim()
            ylim = axs[tag].set_ylim(0.0,ylim[1])
        
        return fig











class UVSpectraRawMixin(object):
    """
    Mixin class for raw UV data to go trhough voigt fitting process
    """

    def firstOfunc(self, x, a, b):
        """
        first order polynomial
        """
        return a*x + b
    def secondOfunc(self, x, a, b, c):
        """
        second order polynomial
        """
        return a*x**2 + b*x + c
    def thirdOfunc(self, x, a, b, c, d):
        """
        third order polynomial
        """
        return a*x**3 + b*x**2 + c*x + d
    def fourthOfunc(self, x, a, b, c, d, e):
        """
        fourth order polynomial
        """
        return a*x**4 + b*x**3 + c*x**2 + d*x + e

    def fifthOfunc(self, x, a, b, c, d, e, f):
        """
        fourth order polynomial
        """
        return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

    def sixthOfunc(self, x, a, b, c, d, e, f, g):
        """
        fourth order polynomial
        """
        return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g


    def filter_regions(self):
        """
        Filters out regions from auto reading in data to exclude "bad" sections 


        """
        # Check data files
        

        to_remove_inds = []
        for ell,region in enumerate(self.dataset.regions):
            try:
                if self.specID_to_suffix[region.specID] not in self.tag_file_pairs[region.lines[0].tag]:
                    to_remove_inds.append(ell)
            except KeyError:
                pass

        if len(to_remove_inds) > 0:
            for ell in np.flip(to_remove_inds):
                _ = self.dataset.regions.pop(ell)




    def get_profile(self, ion_wav, pars, wl_arr = None, vel_arr = None, 
                    resolution = None, redshift = None, **kwargs):
        """
        Get absorption line profile for specified ion_wavelength

        Parameters
        ----------
        ion_wav: `str`
            ion_wavelength to get profile for
        pars: `dict`
            parameters for line profile
            must include "v", "b", and "logN"
        resolution: `number`, optional, must be keyword
            instrument resolution for convolution in km/s
            defualt to 20 km/s
        redshift: `number`, optional, must be keyword
            system redshift, default to 0
        wl_arr: `list-like`, optional, must be keyword
            array of wavelengths to compute spectrum to
        vel_arr: `list_like`, optional, must be keyword
            array of velocities to compute spectrum to
        """
        from VoigtFit.container.lines import show_transitions
        from VoigtFit.funcs.voigt import Voigt, convolve
        from scipy.signal import fftconvolve
        from scipy.signal.windows import gaussian

        # find the right transition
        atomic_data = show_transitions(ion = ion_wav.split("_")[0])
        names = [at[0] for at in atomic_data]
        match = np.array(names) == ion_wav

        _, _, l0, f, gam, _ = np.array(atomic_data)[match][0]

        # check for resolution
        if resolution == None:
            resolution = 20. #km/s for COS

        # check redshift
        if redshift == None:
            redshift = self.redshift

        v = pars["v"]
        b = pars["b"]
        logN = pars["logN"]

        if ((type(v) == float) | (type(v) == np.float64)):
            v = list([v])
            b = list([b])
            logN = list([logN])

        l_center = l0 * (redshift + 1.)

        # find wl_line
        if np.any(vel_arr) == None:
            if np.any(wl_arr) == None:
                # use default wavelength range of +-500 km/s
                wl_arr = np.arange(l_center - .01*250, l_center +0.01*250, 0.01)

            vel_arr = (wl_arr - l_center)/l_center*(speed_of_light.to(u.km/u.s).value)

        elif np.any(wl_arr) == None:
            wl_arr = (vel_arr / (speed_of_light.to(u.km/u.s).value) * l_center) + l_center

        tau = np.zeros_like(vel_arr)

        for (vv, bb, NN) in zip(v, b, logN):
            tau += Voigt(wl_arr, l0, f, 10**NN, 1.e5*bb, gam, z = vv/(speed_of_light.to(u.km/u.s).value))


        # Compute profile
        profile_int = np.exp(-tau)

        # convolve with instrument profile
        if isinstance(resolution, float):
            pxs = np.diff(wl_arr)[0] / wl_arr[0] * (speed_of_light.to(u.km/u.s).value)
            sigma_instrumental = resolution / 2.35482 / pxs
            LSF = gaussian(len(wl_arr) // 2, sigma_instrumental)
            LSF = LSF/LSF.sum()
            profile = fftconvolve(profile_int, LSF, 'same')
        else:
            profile = voigt.convolve_numba(profile_int, resolution)



        out = {
            "wl":wl_arr,
            "vel":vel_arr,
            "spec":profile
        }

        return out

    def plot_region_fit(self, region_ind, sub_region_ind = None, vel_range = None,
                        ax = None, ax_resid = None, figsize = None, labelx = True, 
                        labely = True, comp_scale = None, fit_kwargs = {}, comp_kwargs = {},
                        plot_indiv_comps = False, use_flags = None, ylabel_as_ion = False,
                        **kwargs):
        """
        method to plot a single region fit

        Parameters
        ----------
        region_ind: `int`
            index of region to plot
        sub_region_ind: `int`, optional, must be keyword
            if region has multiple transitions, specifies which index to plot
        vel_range: `list`, optional, must be keyword
            velocity range to plot, defaults ot +/- 500 km/s
        ax: `matplotlib.pyplot.axes`, optional, must be keyword
            axis to plot onto
        ax_resid: `matplotlib.pyplot.axes`, optional, must be keyword
            residual axis to plot onto, if provided
        figsize: `tuple`, optional, must be keyword
            sets figure size if ax not specified
        labelx: `bool`
            if True, labels x axis
        labely: `bool`
            if True, labels y axis
        comp_scale, 'number', optional, must be keyword
            size to make component marker relative to continuum error, default to 1
        fit_kwargs: `dict`
            keywords for fit profile plotting
        comp_kwargs: `dict`
            keywords for vlines marker component velocities

        """
        if self.pre_rebin:
            rebin_n = 1

        if ax is None:
            fig = plt.figure(figsize = figsize)
            gs = plt.GridSpec(8,8)

            ax = fig.add_subplot(gs[1:,:])
            ax_resid = fig.add_subplot(gs[0,:], sharex = ax)

        # Get ion info:
        lines = self.dataset.regions[region_ind].lines

        n_lines = len(lines)
        if n_lines > 1:
            if sub_region_ind == None:
                self._plot_region_fit_sub_ind = 1
                sub_region_ind = 0
            elif (sub_region_ind < (n_lines - 1)):
                self._plot_region_fit_sub_ind = sub_region_ind + 1
            else:
                self._plot_region_fit_sub_ind = None
        else:
            sub_region_ind = 0

        line = lines[sub_region_ind]

        ion = line.ion
        tag = line.tag

        # get params
        try:
            pars = self.dataset.best_fit
        except AttributeError:
            pars = self.dataset.pars

        # extract relevant params
        n_comp = len(self.dataset.components[ion])

        ion_pars = {"v":[], "b":[], "logN":[]}
        for ell in range(n_comp):
            ion_pars["v"].append((pars["z{}_{}".format(ell, ion)].value) * speed_of_light.to(u.km/u.s).value)
            ion_pars["b"].append(pars["b{}_{}".format(ell, ion)].value)
            ion_pars["logN"].append(pars["logN{}_{}".format(ell, ion)].value)


        # Get data
        wl, spec, err, spectra_mask_o = self.dataset.regions[region_ind].unpack()
        if not self.pre_rebin:
            if self.specID_to_suffix[self.dataset.regions[region_ind].specID] != "G160M":
                rebin_n = 5
            else:
                rebin_n = 3
        else:
            rebin_n = 1

        wl_r, spec_r, err_r = rebin_spectrum(wl, spec, err, rebin_n, method = self.rebin_method)
        spectra_mask = rebin_bool_array(spectra_mask_o, rebin_n)

        mask_idx = np.where(spectra_mask == 0)[0]
        mask_idxp1 = mask_idx+1
        mask_idxn1 = mask_idx-1
        big_mask_idx = np.union1d(mask_idxp1[mask_idxp1<(len(spectra_mask)-1)], mask_idxn1[mask_idxn1>0])
        big_mask = np.ones_like(spectra_mask, dtype=bool)
        big_mask[big_mask_idx] = False

        l0_ref, f_ref, _ = line.get_properties()
        l_ref = l0_ref*(self.redshift+1)
        vel = (wl - l_ref) / l_ref * speed_of_light.to(u.km/u.s).value
        vel_r = (wl_r - l_ref) / l_ref * speed_of_light.to(u.km/u.s).value

        profile = self.get_profile(tag, ion_pars, vel_arr = vel, resolution = self.dataset.regions[region_ind].res)
        profile_r = self.get_profile(tag, ion_pars, vel_arr = vel_r, resolution = self.dataset.regions[region_ind].res)

        resid = profile_r["spec"] - spec_r
        cont_err = self.dataset.regions[region_ind].cont_err


        # plot spectra
        # Check kwargs
        if "lw" not in kwargs:
            kwargs["lw"] = 2
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.7
        if "drawstyle" not in kwargs:
            kwargs["drawstyle"] = "steps-mid"
        if "color" not in kwargs:
            kwargs["color"] = "k"

        if "lw" not in fit_kwargs:
            fit_kwargs["lw"] = kwargs["lw"]
        if "alpha" not in fit_kwargs:
            fit_kwargs["alpha"] = kwargs["alpha"]
        # if "drawstyle" not in fit_kwargs:
        #     fit_kwargs["drawstyle"] = kwargs["drawstyle"]
        if "color" not in fit_kwargs:
            fit_kwargs["color"] = "r"

        masked_kwargs = kwargs.copy()
        masked_kwargs["alpha"] *= 0.25



        # plot masked region
        _ = ax.plot(np.ma.masked_where(big_mask, vel_r), 
                    np.ma.masked_where(big_mask, spec_r), 
                    **masked_kwargs)

        if ax_resid != None:

            _ = ax_resid.plot(np.ma.masked_where(big_mask, vel_r), 
                              np.ma.masked_where(big_mask, resid), 
                              **masked_kwargs)

        # plot main region
        _ = ax.plot(np.ma.masked_where(~spectra_mask, vel_r), 
                    np.ma.masked_where(~spectra_mask, spec_r),
                    **kwargs)

        if ax_resid != None:
            _ = ax_resid.plot(np.ma.masked_where(~spectra_mask, vel_r), 
                          np.ma.masked_where(~spectra_mask, resid), 
                          **kwargs)


        if vel_range is None:
            vel_range = ax.set_xlim(-450, 450)
        else:
            vel_range = ax.set_xlim(vel_range)

        
        # plot continuum marker
        _ = ax.axhline(1., ls = "--", lw = 1, color = "k", zorder = -2, alpha = 0.7)
        if ax_resid != None:
            _ = ax_resid.axhline(0., ls = "--", lw = 1, color = "k", zorder = -2, alpha = 0.7)

        # plot continuum error range
        _ = ax.axhline(1+cont_err, ls=":", lw = 1, color = "k", alpha = 0.5, zorder = -2)
        _ = ax.axhline(1-cont_err, ls=":", lw = 1, color = "k", alpha = 0.5, zorder = -2)

        # plot resid error range
        if ax_resid != None:
            _ = ax_resid.fill_between(vel_r, 3*err_r, -3*err_r, color = fit_kwargs["color"], alpha = 0.1)

        # plot fit
        _ = ax.plot(profile["vel"], profile["spec"], **fit_kwargs)

        # lims
        xlim = ax.set_xlim(vel_range)
        ylim = ax.get_ylim()

        # add label
        if not ylabel_as_ion:
            _ = ax.text(xlim[0]+np.abs(np.diff(xlim)*.05), ylim[0]+np.abs(np.diff(ylim)*0.05), 
                        r"{} $\lambda${}".format(tag.split("_")[0], tag.split("_")[1]), 
                        fontsize = 12, 
                        ha = "left", 
                        va = "bottom")

        # axis labels
        if labelx:
            _ = ax.set_xlabel("LSR Velocity (km/s)", fontsize = 12)
        if labely:
            if not ylabel_as_ion:
                _ = ax.set_ylabel("Normalized Flux", fontsize = 12)
            else:
                _ = ax.set_ylabel(r"{} $\lambda${}".format(tag.split("_")[0], tag.split("_")[1]), 
                                  fontsize = 12)

        # set resid ylim
        if ax_resid != None:
            _ = ax_resid.set_ylim(np.nanmin(-7*err_r), np.nanmax(7*err_r))

        # mark components
        if comp_scale == None:
            comp_scale = .1
        for vel in ion_pars["v"]:
            vel_shifted = vel - self.redshift * speed_of_light.to(u.km/u.s).value
            if "alpha" not in comp_kwargs:
                comp_kwargs["alpha"] = kwargs["alpha"]
            if "color" not in comp_kwargs:
                comp_kwargs["color"] = "b"
            if "lw" not in comp_kwargs:
                comp_kwargs["lw"] = 2
            _ = ax.plot([vel_shifted, vel_shifted], [1 - comp_scale, 1 + comp_scale], 
                **comp_kwargs)

        if plot_indiv_comps:

            if use_flags != None:
                color_dict = {
                    "U":"orange",
                    "MC":"blue",
                    "MW":"magenta"
                }
                width_dict = {
                    "U":2,
                    "MC":3,
                }
                alpha_dict = {
                    "MC":1,
                    "U":0.8,
                }
                line_dict = {
                    "BB":"--"
                }
            else:
                color_dict = None
                line_dict = None
                width_dict = None


            for ell in range(n_comp):
                ion_pars = {"v":[], "b":[], "logN":[]}
                ion_pars["v"].append((pars["z{}_{}".format(ell, ion)].value) * speed_of_light.to(u.km/u.s).value)
                ion_pars["b"].append(pars["b{}_{}".format(ell, ion)].value)
                ion_pars["logN"].append(pars["logN{}_{}".format(ell, ion)].value)
                if use_flags != None:
                    if not ("B" in use_flags["{}_{}".format(ell, ion)]) | ("C" in use_flags["{}_{}".format(ell, ion)]):
                        if "MC" in use_flags["{}_{}".format(ell, ion)]:
                            alpha = alpha_dict["MC"]
                            color = color_dict["MC"]
                            width = width_dict["MC"]
                        elif "U" in use_flags["{}_{}".format(ell, ion)]:
                            alpha = alpha_dict["U"]
                            width = alpha_dict["U"]
                            color = color_dict["U"]
                        elif "MW" in use_flags["{}_{}".format(ell, ion)]:
                            color = color_dict["MW"]
                            alpha = 0.6
                            width = 2
                        else:
                            color = "green"
                            width = 2
                            alpha = 0.6


                        if "BB" in use_flags["{}_{}".format(ell, ion)]:
                            ls = line_dict["BB"]
                        else:
                            ls = "-"


                        single_profile = self.get_profile(tag, ion_pars, vel_arr = profile["vel"])
                        single_profile_masked = np.ma.masked_array(single_profile["spec"], 
                                                                    mask = single_profile["spec"] > .99)

                        _ = ax.plot(single_profile["vel"], single_profile_masked, 
                                    alpha = alpha, color = color, lw = width, ls = ls)

                else:
                    single_profile = self.get_profile(tag, ion_pars, vel_arr = profile["vel"])
                    single_profile_masked = np.ma.masked_array(single_profile["spec"], 
                                                                mask = single_profile["spec"] > .99)

                    _ = ax.plot(single_profile["vel"], single_profile_masked, 
                                alpha = 0.6, color = 'b', lw = 1, ls = "-")




        return plt.gcf()





    def plot_all_region_fits(self, fig = None, n_cols = None, 
                             figsize = None, vel_range = None,
                             ratio = None, ylim_lock = None, ylabel_as_ion = False,
                             fit_kwargs = {}, plot_indiv_comps = False, use_flags = None,
                             **kwargs):
        """
        method to plot all region fit

        Parameters
        ----------
        fig: `matplotlib.pyplot.figure`, optional, must be keyword
            Figure to use
        n_cols: `int`, optional, must be keyword 
            number of columns to create
        vel_range: `list`, optional, must be keyword
            velocity range to plot, defaults ot +/- 500 km/s
        figsize: `tuple`, optional, must be keyword
            sets figure size if ax not specified
        ratio: `int`, optional, must be keyword
            sets scaling of main plot to residiual plot, default to 5
        ylim_lock: `list`, optional, must be keyword
            if provided, sets all ylims to match provided values
        fit_kwargs: `dict`
            keywords for fit profile plotting

        """
        # check for figure
        if fig == None:
            fig = plt.figure(figsize = figsize)

        
        if ratio == None:
            ratio = 5
        gs_frame = ratio

        # determine number of plots needed to create
        all_lines = np.concatenate([region.lines for region in self.dataset.regions])
        n_lines = len(all_lines)

        if n_cols == None:
            n_cols = 1

        n_rows = np.ceil(n_lines/n_cols)

        gs_size = int(gs_frame * n_rows + n_rows - 1)
        gs = plt.GridSpec(gs_size,n_cols, hspace=0.)

        

        # make all axes:
        axs = []
        ax_resids = []
        row_counter = 0
        start_counter = 0
        col_counter = 0
        for ell in range(n_lines):
            if np.any(ylim_lock) != None:
                if ell == 0:
                    axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter]))
                    ylim = axs[0].set_ylim(ylim_lock)
                else:
                    axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter], 
                                               sharey = axs[0]))
            else:
                axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter]))
            ax_resids.append(fig.add_subplot(gs[start_counter,col_counter], sharex = axs[ell]))
            ax_resids[ell].axes.get_yaxis().set_visible(False)
            ax_resids[ell].axes.get_xaxis().set_visible(False)



            # add ylabel if necessary
            if not ylabel_as_ion:
                if col_counter == 0:
                    _ = axs[ell].set_ylabel("Normalized Flux")
            else:
                if col_counter == 1:
                    _ = axs[ell].yaxis.tick_right()
                    _ = axs[ell].yaxis.set_label_position("right")

            if row_counter == n_rows-1:
                _ = axs[ell].set_xlabel("LSR Velocity (km/s)")
            row_counter += 1
            if row_counter == n_rows:
                start_counter = 0
                col_counter += 1
            else:
                start_counter += gs_frame+1

        axs[-1].set_xlabel("LSR Velocity (km/s)")

        self._plot_region_fit_sub_ind = None
        plot_counter = 0
        for ell, region in enumerate(self.dataset.regions):
            # plot each spec

            _ = self.plot_region_fit(ell, 
                                     sub_region_ind = self._plot_region_fit_sub_ind, 
                                     ax = axs[plot_counter], 
                                     ax_resid = ax_resids[plot_counter], 
                                     labelx = False, 
                                     labely = ylabel_as_ion, 
                                     vel_range = vel_range, 
                                     plot_indiv_comps = plot_indiv_comps, 
                                     use_flags = use_flags,
                                     fit_kwargs = fit_kwargs,
                                     ylabel_as_ion = ylabel_as_ion,
                                     **kwargs)
            plot_counter += 1

            while self._plot_region_fit_sub_ind != None:
                _ = self.plot_region_fit(ell, 
                                         sub_region_ind = self._plot_region_fit_sub_ind, 
                                         ax = axs[plot_counter], 
                                         ax_resid = ax_resids[plot_counter], 
                                         labelx = False, 
                                         labely = ylabel_as_ion, 
                                         vel_range = vel_range,
                                         plot_indiv_comps = plot_indiv_comps, 
                                         use_flags = use_flags,
                                         fit_kwargs = fit_kwargs,
                                         ylabel_as_ion = ylabel_as_ion,
                                         **kwargs)
                plot_counter += 1



        return fig















    def plot_all_regions(self, figsize = None, velocity = True, continuum = None, **kwargs):
        """
        Plots all spectra from regions of interest

        Parameters
        ----------
        kwargs:
            passed on to ax.plot
        figsize:
            figure size to set
        velocity:
            if True, plots x axis in velocity units
        continuum: `list`, optional, must be keyword
            list of continuum fit ouputs for all regions
            if provided, also plots the fits

        """
        if self.pre_rebin:
            rebin_n = 1
        # check kwargs:
        if "lw" not in kwargs:
            kwargs["lw"] = 2
        if "drawstyle" not in kwargs:
            kwargs["drawstyle"] = "steps-mid"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.8

        # # determine plotting frame
        # if len(self.dataset.regions) % 5 == 0:
        #     fig, axs = plt.subplots(int(len(self.dataset.regions)/5),5, figsize = figsize)
        # elif len(self.dataset.regions) % 4 == 0:
        #     fig, axs = plt.subplots(int(len(self.dataset.regions)/4),4, figsize = figsize)
        # elif len(self.dataset.regions) % 3 == 0:
        #     fig, axs = plt.subplots(int(len(self.dataset.regions)/3),3, figsize = figsize)
        # elif len(self.dataset.regions) % 2 == 0:
        #     fig, axs = plt.subplots(int(len(self.dataset.regions)/2),2, figsize = figsize)
        # else:
        #     fig, axs = plt.subplots(len(self.dataset.regions),1, figsize = figsize)
        fig,axs = plt.subplots(len(self.dataset.regions),1, figsize = figsize)

        if continuum == None:
            plot_cont = False
        else:
            plot_cont = True

        for ell, (region, ax) in enumerate(zip(self.dataset.regions, np.array(axs).flatten())):
            wl = region.wl
            spec = region.flux
            err = region.err
            if not self.pre_rebin:
                if self.specID_to_suffix[region.specID] != "G160M":
                    rebin_n = 5
                else:
                    rebin_n = 3
            wl_r, spec_r, err_r = rebin_spectrum(wl, spec, err, rebin_n, method = self.rebin_method)

            if not velocity:
                xx = wl_r
                ax.plot(wl_r, spec_r, **kwargs)
            else:
                l0_ref, f_ref, _ = region.lines[0].get_properties()
                l_ref = l0_ref*(self.redshift+1)
                vel = (wl_r - l_ref) / l_ref * speed_of_light.to(u.km/u.s)
                ax.plot(vel, spec_r, **kwargs)
                xx = vel
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if plot_cont:
                if not self.pre_rebin:
                    if self.specID_to_suffix[region.specID] != "G160M":
                        rebin_n = 5
                    else:
                        rebin_n = 3
                else:
                    self.rebin_n = 1
                wl_r, spec_r, err_r = rebin_spectrum(region.wl, continuum[ell][0], region.err, 
                                                     rebin_n, method = self.rebin_method)
                ax.plot(xx, spec_r, color = "orange", alpha = 0.8, lw = 1, ls = "--")

            


            if region.label == '':
                region.generate_label()

            _ = ax.text(xlim[0], ylim[0], 
                        r"{}: {}".format(ell,region.label),
                        ha = "left", 
                        va = "bottom", 
                        color = 'k')

            if region.normalized:
                ax.plot(xlim, [1,1], color = "r", lw = 1, alpha = 0.8, zorder = 0, ls = ":")

            xlim = ax.set_xlim(xlim)
            ylim = ax.set_ylim(ylim)

        return fig


    def normalize_region(self, region_ind, 
                         mask = None, 
                         left_mask_region = None, 
                         right_mask_region = None, 
                         func = None, 
                         use_polyfit = True, order = None):
        """
        Normalize Spectrum with polynomial continuum fit
        
        Parameters
        ----------
        region_ind: `int`
            index of region to fit
        mask: `bool array`, optional, must be keyword
            if provided, uses this mask to select continuum region
        left_mask_region: `list`,
            [left,right] mask region for left side continuum
        right_mask_region: `list`,
            [left,right] mask region for right side continuum
        func: `callable`,
            Polynomial function to fit - defaults to Linear 1st order
            
        Returns
        -------
        continuum, continuum_error
        """
        if not use_polyfit:
            from scipy.optimize import curve_fit



            if func == None:
                func = self.firstOfunc
        else:
            if order == None:
                order = 1

        region = self.dataset.regions[region_ind]
        
        if mask is not None:
            fit_wl = region.wl[mask]
            fit_flux = region.flux[mask]
        
        else:
            # set line mask
            left_mask = region.wl < left_mask_region[1]
            left_mask &= region.wl > left_mask_region[0]

            right_mask = region.wl < right_mask_region[1]
            right_mask &= region.wl > right_mask_region[0]

            c_mask = left_mask | right_mask

            fit_wl = region.wl[c_mask]
            fit_flux = region.flux[c_mask]

        if not use_polyfit:
            popt, pcov = curve_fit(func, fit_wl, fit_flux)
            continuum = func(region.wl, *popt)
            e_continuum = np.std(fit_flux - func(fit_wl, *popt))
            
            return continuum, e_continuum / np.median(continuum)
        else:
            z = np.polyfit(fit_wl, fit_flux, order)
            p = np.poly1d(z)
            continuum = p(region.wl)
            e_continuum = np.std(fit_flux - p(fit_wl))
            return continuum, e_continuum / np.median(continuum)

    def normalize_all_regions(self, 
                              masks = None, 
                              left_mask_regions = None, 
                          right_mask_regions = None, 
                          func = None, 
                          apply_all = False, order = None, use_polyfit = False):
        """
        Normalize Spectra with polynomial continuum fit for all regions in dataset
        
        Parameters
        ----------
        masks: `list`.
            [mask,mask, ...] list of masks of continuum
        left_mask_regions: `list`,
            [[left,right], ...] list of mask region for left side continuum
        right_mask_regions: `list`,
            [[left,right], ...] list of mask region for right side continuum
        func: `callable`,
            Polynomial function to fit - defaults to Linear 1st order
        apply_all: `bool`, optional, must be keyword
            if True, applies all normalizations and sets dataset.normalized = True
            
        Returns
        -------
        [[continuum, continuum_error]]
        """


            
        # Check lengths
        n = len(self.dataset.regions)
        
        if masks is None:
        
            if (len(left_mask_regions) != n) | (len(right_mask_regions) != n):
                raise TypeError("Improper number of left or right mask regions provided!")

            output = []
            for ell, (region, left_mask_region, right_mask_region) in enumerate(zip(self.dataset.regions, 
                                                                                    left_mask_regions, 
                                                                                    right_mask_regions)):
                continuum, cont_err = self.normalize_region(ell, left_mask_region, right_mask_region, 
                                                            func = func, order = order, use_polyfit = use_polyfit)
                output.append([continuum, cont_err])
                if apply_all:
                    self.dataset.regions[ell].flux = region.flux / continuum
                    self.dataset.regions[ell].err = region.err / continuum
                    self.dataset.regions[ell].cont_err = cont_err
                    self.dataset.regions[ell].normalized = True

            return output
        else:
            if len(masks) != n:
                raise TypeError("Improper number of continuum masks provided!")
            
            output = []
            for ell, (region, mask) in enumerate(zip(self.dataset.regions, masks)):
                continuum, cont_err = self.normalize_region(ell, mask = mask, func = func, 
                                                            order = order, use_polyfit = use_polyfit)
                output.append([continuum, cont_err])
                if apply_all:
                    self.dataset.regions[ell].flux = region.flux / continuum
                    self.dataset.regions[ell].err = region.err / continuum
                    self.dataset.regions[ell].cont_err = cont_err
                    self.dataset.regions[ell].normalized = True
            
            return output

    def check_for_inactive_components(self, verbose = False, force_clean = False):
        """
        Check that no components for inactive elements are defined
        
        Parameters
        ----------
        verbose: `bool`
        force_clean:`bool'
            if True, removes component for inactive elements
        """
        for this_ion in list(self.dataset.components.keys()):
            lines_for_this_ion = [l.active for l in self.dataset.lines.values() if l.ion == this_ion]

            if np.any(lines_for_this_ion):
                pass
            else:
                if verbose:
                    warn_msg = "\n [WARNING] - Components defined for inactive element: %s"
                    print(warn_msg % this_ion)

                if force_clean:
                    # Remove components for inactive elements
                    self.dataset.components.pop(this_ion)
                    if verbose:
                        print("             The components have been removed.")
                print("")


    def prepare_params(self, apply = True, custom_vel_tie = False):
        """
        Prepare lmfit parameters for fitting process
         
        Parameters
        ----------
        apply: `bool`
            if True, sets pars to dataset
        
        Returns
        -------
        Parameters
        """

        vel_thresh_to_z = 5./ speed_of_light.to(u.km/u.s).value
        
        pars = Parameters()
        pars += self.dataset.static_variables
        for ion in self.dataset.components.keys():
            for n, comp in enumerate(self.dataset.components[ion]):
                ion = ion.replace('*', 'x')
                z, b, logN = comp.get_pars()
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                pars.add(z_name, value=np.float64(z), vary=comp.var_z)
                pars.add(b_name, value=np.float64(b), vary=comp.var_b,
                              min=0.)
                pars.add(N_name, value=np.float64(logN), vary=comp.var_N)
                
        # Check for links
        for ion in self.dataset.components.keys():
            for n, comp in enumerate(self.dataset.components[ion]):
                ion = ion.replace('*', 'x')
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                if comp.tie_z:
                    if custom_vel_tie:
                        pars.add(f"delta_{z_name}", 
                                 value = np.float64(0.), 
                                 vary = True, 
                                 min = np.float64(-vel_thresh_to_z),
                                 max = np.float64(vel_thresh_to_z))
                        pars[z_name].expr = f"{comp.tie_z}+delta_{z_name}"
                    else:
                        pars[z_name].expr = comp.tie_z
                if comp.tie_b:
                    pars[b_name].expr = comp.tie_b
                if comp.tie_N:
                    pars[N_name].expr = comp.tie_N
                    
        # Setup Chebyshev parameters:
        if self.dataset.cheb_order >= 0:
            for reg_num, reg in enumerate(self.dataset.regions):
                if not reg.has_active_lines():
                    continue
                p0 = np.median(reg.flux)
                var_par = reg.has_active_lines()
                if np.sum(reg.mask) == 0:
                    var_par = False
                for cheb_num in range(self.dataset.cheb_order+1):
                    if cheb_num == 0:
                        pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=p0, vary=var_par)
                    else:
                        pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=0.0, vary=var_par)

        if apply:
            self.dataset.pars = pars
        return pars

    def set_spectral_mask(self, region_ind, mask = None, left_right = None):
        """
        Applies mask to spectral section to avoid fitting
        
        Paramters
        ---------
        mask: `list like`
            mask array to avoid fitting to
        left_right: `list`
            [left_wl, right_wl] of region to mask
        """
        if mask is None:
            self.dataset.regions[region_ind].mask = ((self.dataset.regions[region_ind].wl < left_right[1]) & 
                                                     (self.dataset.regions[region_ind].wl > left_right[0]))
        else:
            self.dataset.regions[region_ind].mask = mask
            
        self.dataset.regions[region_ind].mask = ~self.dataset.regions[region_ind].mask

    def set_all_spectral_masks(self, masks = None, left_rights = None):
        """
        Applies all masks to spectral section to avoid fitting
        
        Paramters
        ---------

        mask: `list like`
            [mask, mask, ...] array to avoid fitting to
        left_right: `list`
            [[left_wl, right_wl],[left_wl, right_wl],...] of region to mask
        """
        if masks is None:
            for region_ind, left_right in enumerate(left_rights):
                self.set_spectral_mask(region_ind, left_right = left_right)
        else:
            for region_ind, mask in enumerate(masks):
                self.set_spectral_mask(region_ind, mask = mask)



class CloudyModelMixin(object):
    """
    Mixin class for Cloudy Photoionization models
    """


    def update_distance_grid_command(self, distance_grid_command):
        self.distance_grid_command = distance_grid_command
        self.distance_grid = 10**np.arange(*self.distance_grid_command) * u.cm
        self.distnace_grid = self.distance_grid.to(u.kpc)

    def update_hden_grid_command(self, hden_grid_command):
        self.hden_grid_command = hden_grid_command
        self.hden_grid = 10**np.arange(*self.hden_grid_command) * u.cm**-3



    def read_results(self, input_filename = None, grid_hden = True, grid_metals = True):
        # make sure input filename is there
        if input_filename == None:
            if self.input_filename == None:
                raise ValueError("No input filename attribute set!, Try running get_input_file method first")
        else:
            self.input_filename = input_filename

        # get filename_template
        fn_temp = self.input_filename.split("_input")[0]

        # colden files:
        colden_files = np.sort(glob.glob(f"grid*_{fn_temp}_colden.col"))

        #grid_files
        grd_files = np.sort(glob.glob(f"grid*_{fn_temp}_gridrun.grd"))

        # Get grid parameters
        if grid_hden:
            if not grid_metals:
                t = QTable(names = ["INDEX", "FAILURE", "WARNINGS", "EXIT_CODE", "RANK", "SEQ", "HDEN", "GRID_STR"],
                          dtype=(np.int, np.bool, np.bool, '<U9', np.int, np.int, np.float, '<U9'))
                str_to_bool = {"F":False, "T":True}

                # read lines
                for file in grd_files:
                    with open(file, "r") as f:
                        lines = f.readlines()
                        row = lines[-1].strip("\n").split("\t")
                        row_input = [entry.strip(" ") if entry not in ["F", "T"] else str_to_bool[entry] for entry in row]
                        t.add_row(row_input)

                t["DISTANCE"] = np.ones_like(t["HDEN"])*np.float64(fn_temp.split("_")[2])
                t["STOP_COLN"] = np.ones_like(t["HDEN"])*np.float64(fn_temp.split("_")[4])

            else:
                t = QTable(names = ["INDEX", "FAILURE", "WARNINGS", "EXIT_CODE", "RANK", "SEQ", "HDEN", "METALS", "GRID_STR"],
                          dtype=(np.int, np.bool, np.bool, '<U9', np.int, np.int, np.float, np.float, '<U9'))
                str_to_bool = {"F":False, "T":True}

                # read lines
                for file in grd_files:
                    with open(file, "r") as f:
                        lines = f.readlines()
                        row = lines[-1].strip("\n").split("\t")
                        row_input = [entry.strip(" ") if entry not in ["F", "T"] else str_to_bool[entry] for entry in row]
                        t.add_row(row_input)

                t["DISTANCE"] = np.ones_like(t["HDEN"])*np.float64(fn_temp.split("_")[2])
                t["STOP_COLN"] = np.ones_like(t["HDEN"])*np.float64(fn_temp.split("_")[4])


        # Get coldens
        def species_to_colden_str(string):
            split_str = string.split("+")
            if len(split_str) == 1:
                return f"N_{string}I"
            else:
                ion_number_dict = {
                    "":"II",
                    "2":"III",
                    "3":"IV",
                    "4":"V",
                    "5":"VI"
                }

                return f"N_{split_str[0]}{ion_number_dict[split_str[-1]]}"


        t2 = QTable(names = [*map(species_to_colden_str, self.species)], 
                   dtype = ["float"]*len(self.species), units = ["cm**-2"]*len(self.species))
        # read lines
        for file in colden_files:
            with open(file, "r") as f:
                lines = f.readlines()
                row = lines[-1].strip("\n").split("\t")
                t2.add_row([np.float64(val)*u.cm**-2 for val in row])

        out = hstack([t,t2])
        return out

    def read_all_results(self, input_filenames = None, grid_hden = True, grid_metals = True):
        if input_filenames == None:
            input_filenames = glob.glob("distance*_input.in")

        res = vstack([self.read_results(input_filename = fn, 
                                      grid_hden = grid_hden, 
                                      grid_metals = grid_metals) for fn in input_filenames])

        self.cloudy_results = res
        return res

    def get_measured_coldens(self, data, flag = None, ignore_BB_flag = True):
        """
        Gets measured values from voigtfitting 

        flag: flag to get measuremetns for, default to "MS"
        """
        redshift = data.voigtfit[self.source_name]["LOW"].dataset.redshift
        flags = data.voigtfit_flags[self.source_name]["flags"]
        fit_result_low = data.voigtfit[self.source_name]["LOW"].dataset.best_fit
        fit_result_high = data.voigtfit[self.source_name]["HIGH"].dataset.best_fit
        try:
            fit_result_fuse = data.voigtfit[self.source_name]["FUSE"].dataset.best_fit
        except KeyError:
            fit_result_fuse = None

        if flag == None:
            flag = ["MS", "MSys"]

        meas = QTable(names = ["COMP", "V", "ERR_V", "B", "ERR_B", "N", "ERR_N", "LOWER_LIMIT", "BAD_WIDTH", "COMP_NUM"], 
                      dtype = ["<U9", np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, bool, bool, int], 
                      units = [None, "km/s", "km/s", "km/s", "km/s", "cm**-2", "cm**-2", None, None, None])

        def C_flag(flag_row):
            if "S" in flag_row:
                return True
            else:
                return False

        def BB_flag(flag_row):
            if "BB" in flag_row:
                return True
            else:
                return False
        for key in flags.keys():
            if (np.any([f in flags[key] for f in flag])) & ("B" not in flags[key]) & ("C" not in flags[key]):
                try:
                    row = [key,
                           (fit_result_low["z{}".format(key)].value - redshift)/(redshift+1) * speed_of_light.to(u.km/u.s),
                           fit_result_low["z{}".format(key)].stderr/(redshift+1) * speed_of_light.to(u.km/u.s),
                           fit_result_low["b{}".format(key)].value * u.km/u.s, 
                           fit_result_low["b{}".format(key)].stderr * u.km/u.s, 
                           10**fit_result_low["logN{}".format(key)].value * u.cm**-2,
                           10**fit_result_low["logN{}".format(key)].value * u.cm**-2 * fit_result_low["logN{}".format(key)].stderr * np.log(10),
                           C_flag(flags[key]),
                           BB_flag(flags[key]),
                           int(flags[key][-1].split("c")[-1])]
                except KeyError:
                    try:
                        row = [key,
                               (fit_result_high["z{}".format(key)].value - redshift)/(redshift+1) * speed_of_light.to(u.km/u.s),
                               fit_result_high["z{}".format(key)].stderr/(redshift+1) * speed_of_light.to(u.km/u.s),
                               fit_result_high["b{}".format(key)].value * u.km/u.s, 
                               fit_result_high["b{}".format(key)].stderr * u.km/u.s, 
                               10**fit_result_high["logN{}".format(key)].value * u.cm**-2,
                               10**fit_result_high["logN{}".format(key)].value * u.cm**-2 * fit_result_high["logN{}".format(key)].stderr * np.log(10),
                               C_flag(flags[key]),
                               BB_flag(flags[key]),
                               int(flags[key][-1].split("c")[-1])
                               ]
                    except KeyError:
                        row = [key,
                               (fit_result_fuse["z{}".format(key)].value - redshift)/(redshift+1) * speed_of_light.to(u.km/u.s),
                               fit_result_fuse["z{}".format(key)].stderr(redshift+1) * speed_of_light.to(u.km/u.s),
                               fit_result_fuse["b{}".format(key)].value * u.km/u.s, 
                               fit_result_fuse["b{}".format(key)].stderr * u.km/u.s, 
                               10**fit_result_fuse["logN{}".format(key)].value * u.cm**-2,
                               10**fit_result_fuse["logN{}".format(key)].value * u.cm**-2 * fit_result_fuse["logN{}".format(key)].stderr * np.log(10),
                               C_flag(flags[key]),
                               BB_flag(flags[key]),
                               int(flags[key][-1].split("c")[-1])
                               ]
                meas.add_row(row)

        self.meas = meas
        return meas

    def plot_grid_results(self, stop_neutral_column_density, 
                          cloudy_results = None,
                          target_velocity = None, 
                          velocity_tolerance = None,
                          data = None,
                          cmap = None, 
                          norm = None, 
                          ions = None,
                          flag = None,
                          figsize = None):
        """
        Plots grid results using specified stop column density and target velocity
        
        Parameters
        ----------
        stop_neutral_column_density: `number`
            log of stop column density 
        cloudy_results: `astropy.table.Table`, optional, must be keyword
            Table of Cloudy Results, defaults to self.cloudy_results
        target_velocity: `number`, optional, must be keyword
            target velocity to search for component fit matches
        velocity_tolerance: `number`, optional, must be keyword
            tolerance range of velocity matches, defaults to 20 km/s
        data: `dk_hst_tools.UVSpectra`, optional, must be keyword
            data to get fit results from 
        ions: `list-like`, optional, must be keyword
            list of ions to plot
        flag: `str`, optional, must be keyword
            flag to get fit results for; default to MC
        
        """
        
        if cmap == None:
            cmap = "viridis"
        if norm == None:
            norm = Normalize(vmin = 10, vmax = 90)
        
        sm = cm.ScalarMappable(cmap = cmap, norm = norm)
        
        if cloudy_results == None:
            assert self.cloudy_results != None
            
        else:
            self.cloudy_results = cloudy_results
            
        if ions == None:
            ions = ["HI", "HII", "FeII", "SiII", "SiIII", "SiIV", "CII", "CIV"]
            
        
        # check for stop coln match
        mask = self.cloudy_results["STOP_COLN"] == stop_neutral_column_density
        if np.sum(mask) == 0:
            raise ValueError("stop column density provided not found in results!")
            
        # Identify fit result matches if needed
        if (data is not None) & (target_velocity != None):
            if not hasattr(target_velocity, "unit"):
                target_velocity*= u.km/u.s
            else:
                target_velocity = target_velocity.to(u.km/u.s)
            if velocity_tolerance == None:
                velocity_tolerance = 20*u.km/u.s
            if not hasattr(velocity_tolerance, "unit"):
                velocity_tolerance*= u.km/u.s
            else:
                velocity_tolerance = velocity_tolerance.to(u.km/u.s)
                
            if flag == None:
                flag = "MC"
                
            fit_results = self.get_measured_coldens(data, flag = flag)
            true_values = []
            lower_limit = []
            for ion in ions:
                row_mask = [comp.split("_")[-1] == ion for comp in fit_results["COMP"]]
                if np.sum(row_mask) > 0:
                    vel_mask = [(vel < (target_velocity + velocity_tolerance)) & 
                                (vel > (target_velocity - velocity_tolerance)) for vel in fit_results[row_mask]["V"]]
                    if np.sum(vel_mask) > 0:
                        vel_best = np.argmin(np.abs(fit_results[row_mask][vel_mask]["V"] - target_velocity))
                        true_values.append(np.log10([fit_results[row_mask][vel_mask][vel_best]["N"].value, 
                                            fit_results[row_mask][vel_mask][vel_best]["ERR_N"].value]))
                        lower_limit.append(fit_results[row_mask][vel_mask][vel_best]["LOWER_LIMIT"])
                        
                    else:
                        true_values.append([np.nan, np.nan])
                        lower_limit.append(False)
                else:
                    true_values.append([np.nan, np.nan])
                    lower_limit.append(False)
                        
        distances = np.unique(self.cloudy_results["DISTANCE"])
        
        
        if figsize == None:
            figsize = (9,10)
        fig,axs = plt.subplots(4,2, figsize = figsize)
        
        if true_values == None:
            for ax,ion in zip(axs.flatten(), 
                                    ["HI", "HII", "FeII", "SiII", "SiIII", "SiIV", "CII", "CIV"]):

                for D in distances:
                    color = sm.to_rgba(D)
                    mask2 = self.cloudy_results["DISTANCE"] == D
                    mask2 &= mask
                    ax.plot(self.cloudy_results["HDEN"][mask2], 
                            np.log10(self.cloudy_results[f"N_{ion}"][mask2].value), 
                            color = color, lw = 2, alpha = 0.8, label = f"D = {D} kpc")
                    ax.set_title(ion, fontsize = 12)

                    xlim = ax.set_xlim(-3,0)
                    
        else:
            for ax,ion,truth,ll in zip(axs.flatten(), 
                                    ["HI", "HII", "FeII", "SiII", "SiIII", "SiIV", "CII", "CIV"], 
                                    true_values, 
                                       lower_limit):

                for D in distances:
                    color = sm.to_rgba(D)
                    mask2 = self.cloudy_results["DISTANCE"] == D
                    mask2 &= mask
                    ax.plot(self.cloudy_results["HDEN"][mask2], 
                            np.log10(self.cloudy_results[f"N_{ion}"][mask2].value), 
                            color = color, lw = 2, alpha = 0.8, label = f"D = {D} kpc")
                    
                    xlim = ax.set_xlim(-3,0)
                    
                    if ion == "HI":
                        ax.set_title(f"{ion}; Stop Column Density = {stop_neutral_column_density:.2f}", fontsize = 12)
                        ax.hlines(stop_neutral_column_density, *xlim, color = 'r', lw = 2, ls = '--', alpha = 0.8)
                    else:
                        ax.set_title(ion, fontsize = 12)
                        
                    

                    
                    if not np.isnan(truth[0]):
                        ax.hlines(truth[0], *xlim, color = 'r', lw = 2, ls = '--', alpha = 0.8)
                        ax.fill_between(xlim, 
                                        [truth[0] - truth[1], truth[0] - truth[1]],
                                        [truth[0] + truth[1], truth[0] + truth[1]], 
                                        color = "r", alpha = 0.05)
                        if ll:
                            ax.arrow(-2.5, truth[0], 0, 1.5, width = .005, color = "r", 
                                     zorder = -1, alpha = 0.6, head_width = .05)
                        
            


        lg = ax.legend()

        for ax in axs[3,:]:
            ax.set_xlabel(r"Hydrogen Density (cm$^{{-3}}$)", fontsize = 12)

        for ax in axs[:,0]:
            ax.set_ylabel(r"$\log_{{10}}$ (N / cm$^{{-2}}$)", fontsize = 12)

        for ax in axs[:,1]:
            ax.yaxis.tick_right()

        plt.tight_layout()
        
        return fig
            
        
        
        
        
    
    
                













