import logging

import numpy as np 
from astropy import units as u 
from astropy.table import Table 
from astropy.coordinates import SkyCoord, LSR
from astropy.table.column import (BaseColumn, Column, MaskedColumn, _auto_names, FalseArray,
                     col_copy, _convert_sequence_data_to_array)

from astropy.constants import c as speed_of_light

from .uvDataMixin import UVSpectraMixin, UVSpectraRawMixin, CloudyModelMixin

import os

import glob
import io
import pandas as pd
import sys

from astroquery.simbad import Simbad

import VoigtFit
import pickle

from .JBH_IonizationModel import get_input_spectra

directory = os.path.dirname(__file__)

cite_these = {
    "pymccorrelation":"https://github.com/privong/pymccorrelation",
    "Kendall's Tau with censoring": "https://ui.adsabs.harvard.edu/abs/1986ApJ...306..490I/abstract",
    "pyKrige":"DOI:10.5281/zenodo.3991907",
    "sbi_python":"DOI:10.21105/joss.02505",
    "Simulation Based Inference Review":"https://doi.org/10.1073/pnas.1912789117"
}

def prepare_night_only_data(directory, output_filename = None):
    """
    Prepares night only data for OI using G130_M COS data

    Parameters
    ----------
    directory: `str`
        directory of data
    output_filename: `str`, optional, must be keyword
        name of spectra file to save
    """

    # get source name from directory
    source_name = directory.split("/")[-1]


    # default output file
    if output_filename == None:
        output_filename = f"{source_name}_spec-G130M-N-DK"

    from scipy.interpolate import interp1d

    from calcos import calcos
    from costools import timefilter

    from astroquery.mast import Observations

    from astropy.io import fits

    # Download data
    print(f"Downloading corrtag data for {source_name} from MAST...")
    obs_table = Observations.query_object(source_name, radius = "2 arcmin")
    mask = obs_table["obs_collection"] == "HST"
    mask &= obs_table["instrument_name"] == "COS/FUV"
    mask &= obs_table["filters"] == "G130M"
    target_table = obs_table[mask]
    product_list = Observations.get_product_list(target_table)
    corrtag_mask = ["corrtag" in entry for entry in product_list["dataURI"]]
    products = product_list[corrtag_mask]
    manifest = Observations.download_products(products, download_dir=directory)

    # Get reference files if needed
    print("Checking for reference files...")
    os.system("crds bestrefs --update-bestrefs --sync-references=1 --files {}/mastDownload/HST/*/*.fits".format(directory))

    print("Filtering and processing night only observations...")
    for dataset in glob.glob(f'{directory}/mastDownload/HST/*/*corrtag*.fits'):
        filepath, filename = os.path.split(dataset)
        print ("Filtering ", filename)
        timefilter.TimelineFilter(input=dataset, filter='SUN_ALT > 0')
    for dataset in glob.glob(f'{directory}/mastDownload/HST/*/*corrtag_a*.fits'):
        calcos(dataset, outdir=f'{directory}/nightOnly/')

    def avg_spectra(path):
        wls = []
        fluxs = []
        errs = []
        for dataset in glob.glob(f'{path}/nightOnly/*x1d.fits'):
            with fits.open(dataset) as hdu:
                f = hdu[1].data['flux'].ravel()
                if not np.all(f == 0.):
                    w = hdu[1].data['wavelength'].ravel()
                    wargs = np.argsort(w)
                    wls.append(w[wargs])
                    fluxs.append(f[wargs])
                    errs.append(hdu[1].data['error'].ravel()[wargs])
                
        if len(wls) > 1:
            wl_master = wls[0]
            fs = [fluxs[0]]
            es = [errs[0]]
            
            for wl, f, e in zip(wls[1:], fluxs[1:], errs[1:]):
                interper_f = interp1d(wl, f, bounds_error = False)
                interper_e = interp1d(wl, e, bounds_error = False)
                fs.append(interper_f(wl_master))
                es.append(interper_e(wl_master))
                
            fs = np.vstack(fs)
            es = np.vstack(es)
            
            f_avg = np.nanmean(fs, axis = 0)
            e_avg = np.sqrt(np.nansum(es**2, axis = 0)/es.shape[1])
        elif len(wls) == 1:
            return wls[0], fluxs[0], errs[0]
        else:
            return [0,],[0,],[0,]
            
        return wl_master, f_avg, e_avg

    # Averaging all observations
    print("Averaging all observations if needed...")
    w,f,e = avg_spectra(directory)

    # write to file
    print(f"Saving spectra to file {output_filename}")
    np.savetxt(f"{directory}/{output_filename}", np.stack([w,f,e]).T)



class UVSpectra(UVSpectraMixin, Table):
    """
    Core UV Spectra class

    Load, view, manipulate, and plot basic results from Bart's analysis files

    Parameters
    ----------

    directory: `str`, optional 
        Directory where data is stored, 
        organized as one folder per source/direction
    bart: `bool`, optional, must be keyword
        if True (default) assumes reading in Bart Wakker's data analysis products
    query: `bool`, optional must be keyword
        if True (default) will Simbad query the source names and store their information

    """

    def __init__(self, path = None, bart = True, query = True,
                 source_dirs = None, source_names = None, source_info = None, 
                 LMC_info = None, LMC_coords = None, source_coords = None,
                 coords_dict = None, abund_files = None, 
                 SMC_info = None, SMC_coords = None,
                 raw_df = None, 
                 raw_table = None,
                 voigtfit_files = None,
                 voigtfit = None,
                 voigtfit_flags = None,
                 **kwargs):

        # Read in paths for data

        self.path = path
        if source_dirs is None:
            self.source_dirs = glob.glob(os.path.join(path,"*"))
        else:
            self.source_dirs = source_dirs

        # Get source names:
        if source_names is None:
            self.source_names = [path.split("/")[-1] for path in self.source_dirs]
        else:
            self.source_names = source_names

        # Query basic info to store
        if query:
            if source_info is None:
                self.source_info = Simbad.query_objects(np.unique(self.source_names))
                self.source_info["SOURCE"] = self.source_names
            else:
                self.source_info = source_info

            if LMC_info is None:
                self.LMC_info = Simbad.query_object("LMC")
            else:
                self.LMC_info = LMC_info

            if SMC_info is None:
                self.SMC_info = Simbad.query_object("SMC")
            else:
                self.SMC_info = SMC_info
        

            # Set SkyCoord objects for sources
            if LMC_coords is None:
                self.LMC_coords = SkyCoord(ra = self.LMC_info["RA"], 
                    dec = self.LMC_info["DEC"], 
                    unit = (u.hourangle, u.deg), 
                    frame = "icrs")
            else:
                self.LMC_coords = LMC_coords

            if SMC_coords is None:
                self.SMC_coords = SkyCoord(ra = self.SMC_info["RA"], 
                    dec = self.SMC_info["DEC"], 
                    unit = (u.hourangle, u.deg), 
                    frame = "icrs")
            else:
                self.SMC_coords = SMC_coords


            if source_coords is None:
                self.source_coords = SkyCoord(ra = self.source_info["RA"], 
                    dec = self.source_info["DEC"], 
                    unit = (u.hourangle, u.deg),
                    frame = "icrs")
            else:
                self.source_coords = source_coords

            if coords_dict is None:
                self.coords_dict = {}
                for key,value in zip(self.source_names,self.source_coords):
                    self.coords_dict[key] = value
            else:
                self.coords_dict = coords_dict

        if abund_files is None:
            self.abund_files = glob.glob(os.path.join(self.path,"*/*ABUND.txt"))
        else:
            self.abund_files = abund_files


        # Read in abundance measurements
        names = [
            "SOURCE", 
            "CLASS_FLAG", 
            "NAME", 
            "VELOCITY", 
            "VMIN", 
            "VMAX", 
            "MEAN_VEL", 
            "VEL_GALDEV", 
            "MEASURE_FLAG",
            "N_HI",
            "N_CII", 
            "N_OI", 
            "N_NI", 
            "N_AlII", 
            "N_SiII", 
            "N_SiIII", 
            "N_SII", 
            "N_PII", 
            "N_FeII", 
            "N_OVI",
            "N_CIV", 
            "N_NV", 
            "N_SiIV", 
            "LOG_OVI/CIV", 
            "LOG_CIV/NV", 
            "LOG_CIV/SiIV", 
            "LOG_CIV/II", 
            "LOG_SiIII/II"
        ]

        def read_abund_file(filename, names = names):
            stream = os.popen('cat {} | grep " m "'.format(filename))
            output = stream.read()
            data = io.StringIO(output)
            return  pd.read_csv(data, delim_whitespace=True, names = names, na_values = ".")

        


        if raw_df is None:
            frames = [read_abund_file(abund_file) for abund_file in self.abund_files]
            self.raw_df = pd.concat(frames)
        else:
            self.raw_df = raw_df
        if raw_table is None:
            self.raw_table = Table.from_pandas(self.raw_df)
        


            # Assign Units and convert data when necessary
            vel_cols = ["VELOCITY", "VMIN", "VMAX", "MEAN_VEL", "VEL_GALDEV"]
            for key in vel_cols:
                new_col = self.raw_table[key] * u.km/u.s

                self.raw_table[key] = new_col

            N_cols = [
                "N_HI",
                "N_CII", 
                "N_OI", 
                "N_NI", 
                "N_AlII", 
                "N_SiII", 
                "N_SiIII", 
                "N_SII", 
                "N_PII", 
                "N_FeII", 
                "N_OVI",
                "N_CIV", 
                "N_NV", 
                "N_SiIV", 
            ]

            for species in N_cols:
                upper_limit_mask = [str(val)[0] == "<" for val in self.raw_table[species]]
                lower_limit_mask = [str(val)[0] == ">" for val in self.raw_table[species]]
                new_col = []
                for val in self.raw_table[species]:
                    try:
                        new_col.append(float(str(val).split("<")[-1].split(">")[-1]))
                    except ValueError:
                        new_col.append(np.nan)
                new_col = np.array(new_col)

                self.raw_table[species] = 10**new_col * u.cm**-2
                self.raw_table["{}_UPPERLIMIT".format(species)] = upper_limit_mask
                self.raw_table["{}_LOWERLIMIT".format(species)] = lower_limit_mask



            ratio_cols = [
                "LOG_OVI/CIV", 
                "LOG_CIV/NV", 
                "LOG_CIV/SiIV", 
                "LOG_CIV/II", 
                "LOG_SiIII/II"
            ]

            for ratio in ratio_cols:
                num_species, denom_species = ratio.split("/")
                num_species = num_species.split("_")[-1]
                if denom_species == "II":
                    denom_species = "{}II".format(num_species.split("I")[0])
                self.raw_table[ratio] = np.log10(self.raw_table["N_{}".format(num_species)] / self.raw_table["N_{}".format(denom_species)])
                self.raw_table["{}_UPPERLIMIT".format(ratio)] = self.raw_table["N_{}_UPPERLIMIT".format(num_species)]
                self.raw_table["{}_LOWERLIMIT".format(ratio)] = self.raw_table["N_{}_UPPERLIMIT".format(denom_species)]
                self.raw_table[ratio][self.raw_table["{}_UPPERLIMIT".format(ratio)] & self.raw_table["{}_LOWERLIMIT".format(ratio)]] = np.nan
                self.raw_table["{}_UPPERLIMIT".format(ratio)] |= self.raw_table["N_{}_LOWERLIMIT".format(denom_species)]
                self.raw_table["{}_LOWERLIMIT".format(ratio)] |= self.raw_table["N_{}_LOWERLIMIT".format(num_species)]

                self.raw_table[ratio].unit = None

        else:
            self.raw_table = raw_table



        super().__init__(data = self.raw_table, **kwargs)

        # Add in coordinate information 
        self.SkyCoords = self.get_SkyCoords()
        self.SkyCoords_gal = self.SkyCoords.transform_to("galactic")

        # Add to table
        self["RA"] = self.SkyCoords.ra
        self["DEC"] = self.SkyCoords.dec
        self["GAL-LON"] = self.SkyCoords_gal.l 
        self["GAL-LAT"] = self.SkyCoords_gal.b 

        # Add impact parameters
        self["LMC_ANG_B"] = self.get_angular_impact_parameter(self.LMC_coords, self.SkyCoords)
        self["LMC_B"] = self.get_LMC_impact_parameter(self.SkyCoords)

        # Simple Velocity cut flagging
        self["250_PM_30KMS_FLAG"] = self["MEAN_VEL"] < 280*u.km/u.s
        self["250_PM_30KMS_FLAG"] &= self["MEAN_VEL"] > 220*u.km/u.s

        # check for voigtfit data

        if voigtfit_files == None:
            self.voigtfit_files = glob.glob(os.path.join(self.path,"*","*_VoigtFit_DK.hdf5"))
        else:
            self.voigtfit_files = []

        if voigtfit == None:
            self.voigtfit = {}
            if len(self.voigtfit_files) > 0:
                for f in self.voigtfit_files:
                    sn = f.split("/")[-1].split("_Voigt")[0]
                    self.voigtfit[sn] = UVSpectraRaw(f, from_dataset = True)

        if voigtfit_flags == None:
            self.voigtfit_flags = {}
            if len(self.voigtfit_files) > 0:
                for f in self.voigtfit_files:
                    sn = f.split("/")[-1].split("_Voigt")[0]
                    fn = f.split("/")[:-1]
                    with open("/{}/{}_VoigtFit_Flags_DK.pkl".format(os.path.join(*fn), sn), "rb") as file:
                        self.voigtfit_flags[sn] = pickle.load(file)




    def _new_from_slice(self, slice_):
        """Create a new table as a referenced slice from self."""

        table = self.__class__(masked=self.masked, 
                               path = self.path,
                               source_dirs = self.source_dirs, 
                               source_names = self.source_names, 
                               source_info = self.source_info, 
                               LMC_info = self.LMC_info, 
                               LMC_coords = self.LMC_coords, 
                               source_coords = self.source_coords,
                               coords_dict = self.coords_dict,
                               abund_files = self.abund_files, 
                               raw_df = self.raw_df, 
                               raw_table = self.raw_table, 
                               SMC_info = self.SMC_info, 
                               SMC_coords = self.SMC_coords, 
                               voigtfit_files = self.voigtfit_files,
                               voigtfit = self.voigtfit)
        if self.meta:
            table.meta = self.meta.copy()  # Shallow copy for slice
        table.primary_key = self.primary_key

        newcols = []
        for col in self.columns.values():
            newcol = col[slice_]

            # Note in line below, use direct attribute access to col.indices for Column
            # instances instead of the generic col.info.indices.  This saves about 4 usec
            # per column.
            if (col if isinstance(col, Column) else col.info).indices:
                # TODO : as far as I can tell the only purpose of setting _copy_indices
                # here is to communicate that to the initial test in `slice_indices`.
                # Why isn't that just sent as an arg to the function?
                col.info._copy_indices = self._copy_indices
                newcol = col.info.slice_indices(newcol, slice_, len(col))

                # Don't understand why this is forcing a value on the original column.
                # Normally col.info does not even have a _copy_indices attribute.  Tests
                # still pass if this line is deleted.  (Each col.info attribute access
                # is expensive).
                col.info._copy_indices = True
            else:
                newcol.info.indices = []

            newcols.append(newcol)

        self._make_table_from_cols(table, newcols, verify=False, names=self.columns.keys())
        return table

    def _cite(self):
        return cite_these



class UVSpectraRaw(UVSpectraRawMixin, object):
    """
    Raw UV data reader and wrapper to go through voigt fitting process

    Parameters
    ----------
    filename: `str`, `list-like`
        filename of spectra text file with columns of wavelength, flux, error
        if list, can be multiple filenames of data to load in
    from_dataset: `bool`, optional, must be keyword
        if True, loads from existing saved dataset
    redshift: `number`, optional, must be keyword - defaults to 0
        redshift
    name: `str`, optional, must be keyword
        name to set dataset to, defaults to folder name
    resolution: `number`, optional, must be keyword
        spectral resolution in km/s
    lines: `list-like`, optional, must be keyword
        list of strings of lines to add to dataset
    velspan: `number`, optional, must be keyword
        velocity span for lines, default to 1000
    rebin_n: `number`, optional, must be keyword
        number of elements to rebin by
    rebin_method: `str`, optional, must be keyword
            rebinning method to use, either "mean" or "median"
    """
    
    def __init__(self, filename,
                 from_dataset = False, 
                 redshift = None, 
                 name = None, 
                 resolution = None, 
                 lines = None, 
                 velspan = None, 
                 rebin_n = None,
                 rebin_method = None, 
                 query = True):

        if not from_dataset:
            if filename.__class__ is str:
                self.data_files = [filename]
            else:
                self.data_files = filename
            

            if resolution == None:
                self.resolution = 20. # COS

            if name == None:
                self.name = self.data_files[0].split("/")[-2]
            else:
                self.name = name

            if lines == None:
                # set default set of lines
                self.lines = lines = ["CII_1334", 
                         "CIV_1548", "CIV_1550",
                         "SiII_1190", "SiII_1193", "SiII_1260", 
                         "SiIII_1206", 
                         "SiIV_1393", "SiIV_1402",]
                         # "OI_1302",
                         # "OVI_1031", "OVI_1037",
                         # "NV_1238", "NV_1242",
                         # "SII_1250", "SII_1253"]#, "SII_1259"]
            else:
                self.lines = lines

            if velspan == None:
                self.velspan = 500.
            else:
                self.velspan = velspan

            if rebin_n == None:
                self.rebin_n = 5
            else:
                self.rebin_n = rebin_n

            if rebin_method == None:
                self.rebin_method = "mean"
            else:
                self.rebin_method = rebin_method

            customSimbad = Simbad()
            customSimbad.add_votable_fields("rvz_radvel", "rvz_type")

            self.source_info = customSimbad.query_object(self.name)

            self.SkyCoord_at_LMC = SkyCoord(ra = self.source_info["RA"][0], 
                        dec = self.source_info["DEC"][0], 
                        distance = 50*u.kpc,
                        pm_ra_cosdec = 0*u.mas/u.s,
                        pm_dec = 0*u.mas/u.s,
                        radial_velocity = 0*u.km/u.s,
                        unit = (u.hourangle, u.deg), 
                        frame = "icrs")


            self.redshift_from_rv =self.SkyCoord_at_LMC.transform_to(LSR()).radial_velocity/speed_of_light
            self.redshift_from_rv = -1*self.redshift_from_rv.decompose().value

            if redshift == None:
                self.redshift = self.redshift_from_rv
            else:
                self.redshift = redshift

            # open Dataset
            self.dataset = VoigtFit.DataSet(self.redshift)
            self.dataset.set_name(self.name)


            # read in data from text file
            for file in self.data_files:
                print("Loading data from file, {}".format(file.split("/")[-1]))
                try:
                    wav, flux, err, _,_, _,_, _,_ = np.loadtxt(file, unpack = True)
                except ValueError:
                    wav, flux, err = np.loadtxt(file, unpack = True)
                mask = flux < 0
                mask |= np.isnan(flux)
                mask |= np.isinf(flux)
                self.dataset.add_data(wav[~mask], flux[~mask], self.resolution, 
                                      err = err[~mask], 
                                      normalized = False)


            # Add relevent lines to dataset
            for line in self.lines:
                self.dataset.add_line(line, velspan = self.velspan)

        else:
            #loading from existing dataset
            self.data_files = [filename]

            if resolution == None:
                self.resolution = 20.
            else:
                self.resolution = resolution

            # load dataset
            self.dataset = VoigtFit.load_dataset(filename)

            if name == None:
                self.name = self.dataset.name
            else:
                self.name = name

            regions = self.dataset.regions
            self.lines = []
            for region in regions:
                self.lines = np.concatenate([self.lines, [line.tag for line in region.lines]])

            self.velspan = self.dataset.velspan

            if rebin_n == None:
                self.rebin_n = 5
            else:
                self.rebin_n = rebin_n

            if rebin_method == None:
                self.rebin_method = "mean"
            else:
                self.rebin_method = rebin_method

            customSimbad = Simbad()
            customSimbad.add_votable_fields("rvz_radvel", "rvz_type")

            self.source_info = customSimbad.query_object(self.name)

            self.SkyCoord_at_LMC = SkyCoord(ra = self.source_info["RA"][0], 
                        dec = self.source_info["DEC"][0], 
                        distance = 50*u.kpc,
                        pm_ra_cosdec = 0*u.mas/u.s,
                        pm_dec = 0*u.mas/u.s,
                        radial_velocity = 0*u.km/u.s,
                        unit = (u.hourangle, u.deg), 
                        frame = "icrs")


            self.redshift_from_rv =self.SkyCoord_at_LMC.transform_to(LSR()).radial_velocity/speed_of_light
            self.redshift_from_rv = -1*self.redshift_from_rv.decompose().value

            self.redshift = self.dataset.redshift

    def save_dataset(self, filename, in_same_folder = False):
        if not in_same_folder:
            self.dataset.save(filename)
        else:
            path = os.path.join("/",*self.data_files[0].split("/")[:-1], filename)
            self.dataset.save(path)




class CloudyModel(CloudyModelMixin, object):
    """
    Raw UV data reader and wrapper to go through voigt fitting process

    Parameters
    ----------
    source_name: `str`
        Name of source
    source_info: `astropy.table.Table`, optional, must be keyword
        table of source info from Simbad query
    source_coord: `astropy.coordinates.SkyCoord`, optional, must be keyword
        Coordinate of source
        if not provided, will query Simbad to get it
    distance_grid_command: `list-like`, optional, must be keyword
        [min,max,step] of distances to grid in Cloudy in log10 space and units of cm
    hden_grid_command: `list-like`, optional, must be keyword
        [min,max,step] of distances to grid in Cloudy of hydrogen column density in
        log10 space and units of cm^-3
    neutral_column_density_grid: `list-like`, optional, must be keyword
        log10 neutral column densities grid
    spectra_template_filename: `str`, optional, must be keyword
        template tabulated spectrum, defaults to that from Fox et al. 2005 (Figure 8)
    egb: `str`, optional, must be keyword
        extragalactic background spectra to use, default to KS18 
    ebg_redshift: `number`, optional, must be keyword
        redshift to use for extragalactic background, default to 0
    metalicity: `number`, optional, must be keyword
        metalicity to use in log space relative to solar, default to -0.3
    """
    
    def __init__(self, source_name, source_coord = None, source_info = None, 
                 distance_grid_command = None, 
                 neutral_column_density_grid = None, 
                 spectra_template_filename = None, 
                 egb = None, 
                 egb_redshift = None, 
                 cosmic_rays_background = True,
                 species = None,
                 metalicity = None,
                 hden_grid_command = None,):

        self.source_name = source_name

        if source_info == None:
            self.source_info = Simbad.query_object(self.source_name)

        if source_coord is None:
            self.source_coord = SkyCoord(ra = self.source_info["RA"], 
                    dec = self.source_info["DEC"], 
                    unit = (u.hourangle, u.deg), 
                    frame = "icrs").transform_to('galactic')
        else:
            self.source_coord = source_coord.transform_to('galactic')

        if distance_grid_command == None:
            self.distance_grid_command = [22.9,23.55,0.05]
        else:
            self.distance_grid_command = distance_grid_command

        self.distance_grid = 10**np.arange(*self.distance_grid_command) * u.cm
        self.distnace_grid = self.distance_grid.to(u.kpc)


        self.neutral_column_density_grid = neutral_column_density_grid

        if spectra_template_filename != None:
            self.spectra_template_filename = spectra_template_filename
        else:
            self.spectra_template_filename = os.path.join(directory,"data/JBH_RadiationField/Fox+2005_MW.sed")


        if egb == None:
            self.egb = "KS18"
        else:
            self.egb = egb

        if egb_redshift == None:
            self.egb_redshift = 0.
        else:
            self.egb_redshift = egb_redshift

        self.cosmic_rays_background = cosmic_rays_background

        if species == None:
            self.species = ["H", "H+", "Si+", "Si+2", "Si+3", "C+", "C+3", "Fe+"]
        else:
            self.species = species


        if metalicity == None:
            self.metalicity = -0.3
        else:
            self.metalicity = metalicity

        if hden_grid_command == None:
            self.hden_grid_command = [-5,-1,0.5]
        else:
            self.hden_grid_command = hden_grid_command

        self.hden_grid = 10**np.arange(*self.hden_grid_command) * u.cm**-3

        self.input_filename = None


        


    def get_input_file(self, stop_neutral_column_density, distance = None, save = False):
        """
        Returns string of input file as list for each line of file
        """ 

        if distance == None:
            distance = 50*u.kpc

        coord_3d = SkyCoord(l = self.source_coord.l, 
                            b = self.source_coord.b, 
                            distance = distance, 
                            frame = "galactic")

        rad, norms = get_input_spectra(coord_3d)

        input_spectra_filename = self.spectra_template_filename.split("/")[-1]
        #see if file is available in local directory
        if not len(glob.glob(input_spectra_filename))>0:
            import shutil
            shutil.copy(self.spectra_template_filename, "./")


        file_lines = []


        file_lines.append(f'title {self.source_name}')
        file_lines.append('# Input Spectrum File')
        file_lines.append(f'table SED "{input_spectra_filename}"')
        file_lines.append('# Normalization')
        file_lines.append(f'phi(H) = {np.log10(norms["TOTAL"].value)}')
        file_lines.append('# Extragalactic Background')
        file_lines.append(f'Table {self.egb} redshift {self.egb_redshift}')
        file_lines.append('# hden')
        file_lines.append('hden -3.0 vary')
        file_lines.append('grid {} {} {}'.format(*self.hden_grid_command))

        if self.cosmic_rays_background:
            file_lines.append('cosmic rays background')

        file_lines.append('constant density')
        file_lines.append('# Metalcity')
        file_lines.append(f'metals {self.metalicity} log')
        file_lines.append('# Stop condition')
        file_lines.append(f'stop netural column density {stop_neutral_column_density}')

        file_lines.append('double optical depths')
        file_lines.append('iterate to convergence')
        file_lines.append('save grid separate "distance_kpc_{0:.1f}_stopNHI_{1}_gridrun.grd"'.format(distance.value, 
                                                            stop_neutral_column_density))
        file_lines.append('save species column densities last separate "distance_kpc_{0:.1f}_stopNHI_{1}_colden.col" no hash'.format(distance.value, 
                                                            stop_neutral_column_density))
        for species in self.species:
            file_lines.append(f'"{species}"')

        file_lines.append('end of species')
        file_lines.append('print last')
        file_lines.append('plot continuum')

        if save:

            with open('distance_kpc_{0:.1f}_stopNHI_{1}_input.in'.format(distance.value, 
                                                                stop_neutral_column_density), 'w') as f:
                for line in file_lines:
                    print(line, file = f)

            self.input_filename = 'distance_kpc_{0:.1f}_stopNHI_{1}_input.in'.format(distance.value, 
                                                                stop_neutral_column_density)

        return file_lines

    def print_input_file(self, file_lines):
        for line in file_lines:
            print(line)




























