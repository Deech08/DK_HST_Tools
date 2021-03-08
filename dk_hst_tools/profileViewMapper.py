# Interactive plotting to show up to 15 voigtfit lines, HI spectra, and HI map with points for sightlines

from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from spectral_cube import SpectralCube 

from astropy.coordinates import SkyCoord

import os




class ProfileViewMapper():
    """
    Class to interactively plot HST, HI spectra from HI maps and nearest HST sightlines

    Parameters
    ----------
    
    data: `uvSpectra`
    fig: `matplotlib.pyplot.figure`, optional, must be keyword
        Figure to use
    figsize: tuple, optional, must be keyword
        size of figure to create if fig not specified
    cube: `spectral_cube.spectral_cube.SpectralCube`. optional ,must be keyword
        data cube to plot map from. if not provided, loads default map for DK
    vel_range: `list-like`, optional, must be keyword
        velocity range to integrate map
    clip_mask: `bool`, optional, must be keyword
        if True, clips data at clip_value
    clip_value: `number, astropy.units.quantity`, optional, must be keyword
        value to clip intensity at, defaults to 10^18 cm^-2
    """


    def __init__(self, data, fig = None, figsize = None, cube = None, 
                 vel_range = None, clip_mask = True, clip_value = None, 
                 cmap = None, norm = None, hi_spec_kwargs = {}, add_impact_parameter = True):

        self.data = data


        if clip_value is None:
            self.clip_value = 1e18* u.cm**-2
        else:
            self.clip_value = clip_value
        if not hasattr(self.clip_value, "unit"):
            self.clip_value *= u.cm**-2
            logging.warning("No units for clip_value, provided, assuming u.cm^-2")


        # Get data cube
        if cube == None:
            #load custom_cut_cube
            self.cube = os.path.join(self.data.path, "../HI/hi4pi_LMC_full_velocity.fits")
        else:
            self.cube = cube
        if self.cube.__class__ is str:
            self.cube = SpectralCube.read(self.cube)

        # set vel range for map
        if vel_range == None:
            self.vel_range = [150,350] * u.km/u.s
        else:
            self.vel_range = vel_range
        if not hasattr(self.vel_range, "unit"):
            self.vel_range *= u.km/u.s
            logging.warning("No units for vel_range provided, assuming u.km/u.s")

        # Get moment map set-up
        col_den_factor = 1.823*10**18 * u.cm**-2 / (u.K * u.km/u.s)
        self.nhi = self.cube.spectral_slab(*self.vel_range).moment(order = 0)*col_den_factor
        self.nhi = self.nhi.to(u.cm**-2)

        if clip_mask:

            self.masked_moment = np.ma.masked_array(self.nhi.value, mask = self.nhi < self.clip_value)
            
        else:
            self.masked_moment = np.ma.masked_array(self.nhi.value, mask = np.isnan(self.nhi))

        if fig == None:
            if figsize == None:
                figsize = (9,20)
            self.fig = plt.figure(figsize = figsize)

        else:
            self.fig = fig

        # Setup GridSpec

        # Map axis
        self.gs = plt.GridSpec(62,32,hspace = 0.)

        # Control Buttons
        self.next_ax = self.fig.add_subplot(self.gs[:3,11:21])
        self.back_ax = self.fig.add_subplot(self.gs[:3,:10], sharex = self.next_ax, sharey = self.next_ax)
        self.next_ax.set_xticks([])
        self.next_ax.set_yticks([])

        xc = np.median(self.next_ax.get_xlim())
        yc = np.median(self.next_ax.get_ylim())
        _ = self.next_ax.text(xc, yc, "Next\nSource", fontsize = 20, fontweight = "bold", 
                              ha = 'center', va = 'center')
        _ = self.back_ax.text(xc, yc, "Previous\nSource", fontsize = 20, fontweight = "bold", 
                              ha = 'center', va = 'center')


        self.image_ax = self.fig.add_subplot(self.gs[4:26,:20], projection = self.nhi.wcs)

        # plot HI Map
        if cmap == None:
            cmap = "Blues"
        if norm == None:
            norm = LogNorm(vmin = self.clip_value.to(u.cm**-2).value, 
                                             vmax = 1e22)
        im = self.image_ax.imshow(self.masked_moment, cmap = cmap, norm = norm)

        if add_impact_parameter:
            _ = self.data.plot_impact_parameter_contour(ax = self.image_ax, colors = "k", 
                                                        alpha = 0.8, linestyles = ":")

        # add scatter points
        self.c_gal = self.data.source_coords.transform_to('galactic')

        _ = self.image_ax.scatter(self.c_gal.l, self.c_gal.b, 
                                   transform = self.image_ax.get_transform("world"), 
                                   s = 50, color = "k", alpha = 0.3)


        self.hi_ax = self.fig.add_subplot(self.gs[:8,22:])
        # hi spectrum to start

        if "lw" not in hi_spec_kwargs:
            hi_spec_kwargs["lw"] = 1
        if "color" not in hi_spec_kwargs:
            hi_spec_kwargs["color"] = "b"
        if "alpha" not in hi_spec_kwargs:
            hi_spec_kwargs["alpha"] = 0.8

        self.hi_spec_kwargs = hi_spec_kwargs

        self.hi_coord = self.data.LMC_coords
        

        self.hst_axs = [] 
        self.hst_axs.append(self.fig.add_subplot(self.gs[9:17,22:], sharex = self.hi_ax))
        self.hst_axs.append(self.fig.add_subplot(self.gs[18:26,22:], sharex = self.hi_ax, 
                                                 sharey = self.hst_axs[0]))

        self.hst_axs[0].yaxis.tick_right()
        self.hst_axs[1].yaxis.tick_right()
        self.hst_axs[0].set_ylim(0, 1.15)


        # Rest of hst_axs:
        self.hst_axs.append(self.fig.add_subplot(self.gs[27:35,:10], sharex = self.hi_ax))
        self.hst_axs.append(self.fig.add_subplot(self.gs[36:44,:10], sharex = self.hi_ax))
        self.hst_axs.append(self.fig.add_subplot(self.gs[45:53,:10], sharex = self.hi_ax))
        self.hst_axs.append(self.fig.add_subplot(self.gs[54:,:10], sharex = self.hi_ax))

        self.hst_axs.append(self.fig.add_subplot(self.gs[27:35,11:21], sharex = self.hi_ax))
        self.hst_axs[-1].set_yticks([])
        self.hst_axs.append(self.fig.add_subplot(self.gs[36:44,11:21], sharex = self.hi_ax))
        self.hst_axs[-1].set_yticks([])
        self.hst_axs.append(self.fig.add_subplot(self.gs[45:53,11:21], sharex = self.hi_ax))
        self.hst_axs[-1].set_yticks([])
        self.hst_axs.append(self.fig.add_subplot(self.gs[54:,11:21], sharex = self.hi_ax))
        self.hst_axs[-1].set_yticks([])

        self.hst_axs.append(self.fig.add_subplot(self.gs[27:35,22:], sharex = self.hi_ax))
        self.hst_axs[-1].yaxis.tick_right()
        self.hst_axs.append(self.fig.add_subplot(self.gs[36:44,22:], sharex = self.hi_ax))
        self.hst_axs[-1].yaxis.tick_right()
        self.hst_axs.append(self.fig.add_subplot(self.gs[45:53,22:], sharex = self.hi_ax))
        self.hst_axs[-1].yaxis.tick_right()
        self.hst_axs.append(self.fig.add_subplot(self.gs[54:,22:], sharex = self.hi_ax))
        self.hst_axs[-1].yaxis.tick_right()

        self.axes_dict = {
            "FeII_1608":self.hst_axs[0],
            "FeII_1144":self.hst_axs[1],
            "CII_1334":self.hst_axs[2],
            "SiII_1190":self.hst_axs[3],
            "SiII_1193":self.hst_axs[4],
            "SiII_1260":self.hst_axs[5],
            "SiII_1526":self.hst_axs[6],
            "SiIII_1206":self.hst_axs[7],
            "SiIV_1393__0":self.hst_axs[8],
            "SiIV_1393__1":self.hst_axs[9],
            "SiIV_1402__0":self.hst_axs[10],
            "SiIV_1402__1":self.hst_axs[11],
            "CIV_1548":self.hst_axs[12],
            "CIV_1550":self.hst_axs[13]
        }


        # starting source
        self.source_ind = 0
        
        self.update_for_source()
        self.plot_hi_spec()

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def get_hi_spectrum(self):
        ds9_str = 'galactic; circle({}, {}, 200")'.format(self.hi_coord.transform_to("galactic").l.value[0], 
                                                          self.hi_coord.transform_to("galactic").b.value[0])
        subcube = self.cube.subcube_from_ds9region(ds9_str)
        return np.nanmean(subcube.unmasked_data[:,:,:], axis = (1,2))

    # Step through each region and plot fits
    def plot_fits(self):
        # step through regions data.voigtfit["PKS0637-75"].dataset.regions[0].lines[0].tag
        n_regions = len(self.voigtfit.dataset.regions)
        repeat = False
        SiIV_0 = False
        SiIV_1 = False
        for ell in range(n_regions):
            # get tag
            lines = self.voigtfit.dataset.regions[ell].lines
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

                if tag != "SiII_1304":
                    ax = self.axes_dict[tag]
                    ax.clear()

                    _ = self.voigtfit.plot_region_fit(ell, sub_region_ind = sub_region_ind, 
                                                    vel_range = None,
                        ax = ax, labelx = False, labely = False, lw = 1, alpha = 0.6, 
                                                fit_kwargs = {"lw":1, "alpha":0.5}, 
                                                comp_scale = .2, plot_indiv_comps = True, 
                                                use_flags = self.flags["flags"])

    def plot_hi_spec(self, clear = True):
        if clear:
            self.hi_ax.clear()
        self.hi_spec_data = self.get_hi_spectrum()
        self.hi_vel = self.cube.spectral_axis.to(u.km/u.s)
        
        if clear:
            self.hi_line, = self.hi_ax.plot(self.hi_vel, self.hi_spec_data, **self.hi_spec_kwargs)

            self.hi_ax.yaxis.tick_right()
            self.hi_ax.set_yscale("symlog")
            self.hi_ax.set_title("l = {0:.1f}, b = {1:.1f}".format(self.hi_coord.transform_to('galactic').l.value[0],
                                                         self.hi_coord.transform_to('galactic').b.value[0] ), 
                                 fontsize = 12)
            _ = self.hi_ax.set_xlim(-600,600)
        else:
            _ = self.hi_ax.plot(self.hi_vel, self.hi_spec_data, lw = 1, color = "r", ls = ":", alpha = 0.4)

    def update_for_source(self):
        self.source_name = self.data.source_names[self.source_ind]
        self.voigtfit = self.data.voigtfit[self.source_name]
        self.flags = self.data.voigtfit_flags[self.source_name]

    

        #extra marker for current source
        if hasattr(self, "source_scatter_point"):
            self.source_scatter_point.remove()
        self.source_scatter_point = self.image_ax.scatter(self.c_gal.l[self.source_ind], 
                                                           self.c_gal.b[self.source_ind], 
                                                           transform = self.image_ax.get_transform("world"), 
                                                           s = 50, color = "r", alpha = 0.7, 
                                                           edgecolor = "k")

        # Title
        self.image_ax.set_title("Current Source: {}".format(self.source_name), fontsize = 12)

        

        self.plot_fits()

        self.hi_coord = SkyCoord(l = [self.c_gal.l[self.source_ind]], b = [self.c_gal.b[self.source_ind]], 
                                 frame = "galactic")

        self.plot_hi_spec()

    def on_click(self, event):
        if event.button == 1: # left mouse click
            if event.inaxes is self.next_ax:
                self.source_ind += 1
                if self.source_ind >= len(self.data.source_names):
                    self.source_ind = 0
                self.update_for_source()
            elif event.inaxes is self.back_ax:
                self.source_ind -= 1
                if self.source_ind < 0:
                    self.source_ind = len(self.data.source_names)-1
                self.update_for_source()

            elif event.inaxes is self.image_ax:
                # add more lines on top of HI spectrum
                # Covert coordinates from pixel to world
                lon = event.xdata
                lat = event.ydata
                lon, lat = self.image_ax.wcs.wcs_pix2world(lon, lat, 0)
                # Create SKyCoord
                self.hi_coord = SkyCoord(l = [lon]*u.deg, b = [lat]*u.deg, frame = 'galactic')
                self.plot_hi_spec()


        if event.button == 3: # right mouse click
            if event.inaxes is self.image_ax:
                # add more lines on top of HI spectrum
                # Covert coordinates from pixel to world
                lon = event.xdata
                lat = event.ydata
                lon, lat = self.image_ax.wcs.wcs_pix2world(lon, lat, 0)
                # Create SKyCoord
                self.hi_coord = SkyCoord(l = [lon]*u.deg, b = [lat]*u.deg, frame = 'galactic')
                ylim = self.hi_ax.get_ylim()
                self.plot_hi_spec(clear = False)
                self.hi_ax.set_ylim(ylim)









