import logging

import astropy.units as u
from astropy.coordinates import SkyCoord, concatenate
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm

from pymccorrelation import pymccorrelation

from pykrige.uk import UniversalKriging

import os

from spectral_cube import SpectralCube


from .sbiKit import sbiKit

from lmfit import Parameters

from VoigtFit.output import rebin_spectrum



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

        return sbiKit(self, ion, mask_limits = mask_limits, from_file = from_file)








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

    def plot_all_regions(self, figsize = None, **kwargs):
        """
        Plots all spectra from regions of interest

        Parameters
        ----------
        kwargs:
            passed on to ax.plot
        figsize:
            figure size to set
        

        """

        # check kwargs:
        if "lw" not in kwargs:
            kwargs["lw"] = 2
        if "drawstyle" not in kwargs:
            kwargs["drawstyle"] = "steps-mid"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.8

        # determine plotting frame
        if len(self.dataset.regions) % 5 == 0:
            fig, axs = plt.subplots(int(len(self.dataset.regions)/5),5, figsize = figsize)
        elif len(self.dataset.regions) % 4 == 0:
            fig, axs = plt.subplots(int(len(self.dataset.regions)/4),4, figsize = figsize)
        elif len(self.dataset.regions) % 3 == 0:
            fig, axs = plt.subplots(int(len(self.dataset.regions)/3),3, figsize = figsize)
        elif len(self.dataset.regions) % 2 == 0:
            fig, axs = plt.subplots(int(len(self.dataset.regions)/2),2, figsize = figsize)
        else:
            fig, axs = plt.subplots(len(self.dataset.regions),1, figsize = figsize)

        for ell, (region, ax) in enumerate(zip(self.dataset.regions, np.array(axs).flatten())):
            wl = region.wl
            spec = region.flux
            err = region.err
            wl_r, spec_r, err_r = rebin_spectrum(wl, spec, err, self.rebin_n, method = self.rebin_method)
            ax.plot(wl_r, spec_r, **kwargs)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

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
                         func = None):
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
        from scipy.optimize import curve_fit

        if func == None:
            func = self.firstOfunc

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

        
        popt, pcov = curve_fit(func, fit_wl, fit_flux)
        continuum = func(region.wl, *popt)
        e_continuum = np.std(fit_flux - func(fit_wl, *popt))
        
        return continuum, e_continuum / np.median(continuum)

    def normalize_all_regions(self, 
                              masks = None, 
                              left_mask_regions = None, 
                          right_mask_regions = None, 
                          func = None, 
                          apply_all = False):
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
                continuum, cont_err = self.normalize_region(ell, left_mask_region, right_mask_region, func = func)
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
                continuum, cont_err = self.normalize_region(ell, mask = mask, func = func)
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


    def prepare_params(self, apply = True):
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
        
        pars = Parameters()
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
                set_spectral_mask(self.dataset.regions[region_ind], left_right = left_right)
        else:
            for region_ind, mask in enumerate(masks):
                set_spectral_mask(self.dataset.regions[region_ind], mask = mask)





