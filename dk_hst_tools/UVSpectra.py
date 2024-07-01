import logging

import numpy as np 
from astropy import units as u 
from astropy.table import Table, vstack
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

from VoigtFit.io.output import rebin_spectrum, rebin_bool_array

from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import cm as cmapper
from matplotlib import cm

import warnings

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
    if output_filename is None:
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
            
            f_sum = np.nansum(fs, axis = 0)
            e_sum = np.nansum(es, axis = 0)
        elif len(wls) == 1:
            return wls[0], fluxs[0], errs[0]
        else:
            return [0,],[0,],[0,]
            
        return wl_master, f_sum, e_sum

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.source_info = vstack([Simbad.query_objects([sn]) for sn in self.source_names])
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

        if voigtfit_files is None:
            self.voigtfit_files = {"LOW":glob.glob(os.path.join(self.path,"*","*_VoigtFit_DK_vSeparate2_Low.hdf5")),
                                   "HIGH":glob.glob(os.path.join(self.path,"*","*_VoigtFit_DK_vSeparate2_High.hdf5")),
                                   "FUSE":glob.glob(os.path.join(self.path,"*","*_VoigtFit_DK_FUSE_OVI_v2.hdf5"))}
        else:
            self.voigtfit_files = voigtfit_files

        if voigtfit is None:
            self.voigtfit = {}
            if len(self.voigtfit_files["LOW"]) > 0:
                for fl in self.voigtfit_files["LOW"]:
                    sn = fl.split("/")[-1].split("_Voigt")[0]
                    self.voigtfit[sn] = {"LOW":UVSpectraRaw(fl, from_dataset = True)}
                for fh in self.voigtfit_files["HIGH"]:
                    sn = fh.split("/")[-1].split("_Voigt")[0]
                    self.voigtfit[sn]["HIGH"]=UVSpectraRaw(fh, from_dataset = True)
                for ff in self.voigtfit_files["FUSE"]:
                    sn = ff.split("/")[-1].split("_Voigt")[0]
                    self.voigtfit[sn]["FUSE"]=UVSpectraRaw(ff, from_dataset = True)


        if voigtfit_flags is None:
            self.voigtfit_flags = {}
            if len(self.voigtfit_files["LOW"]) > 0:
                for f in self.voigtfit_files["LOW"]:
                    sn = f.split("/")[-1].split("_Voigt")[0]
                    fn = f.split("/")[:-1]
                    with open("/{}/{}_VoigtFit_Flags_DK_vSeparate2_LowHigh.pkl".format(os.path.join(*fn), sn), "rb") as file:
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
                 query = True,
                 filter_regions = True, 
                 auto_resolution = True,
                 pre_rebin = True, 
                 manual_night_only_error = None, 
                 use_DK_N = True, 
                 shift_night_only_at_OI = None,
                 shift_g160_at_1526 = None, 
                 fuse_only = False,
                 query_name = None,
                 spec_file = None):

        if not from_dataset:
            if filename.__class__ is str:
                self.data_files = [filename]
            else:
                self.data_files = filename

            self.pre_rebin = pre_rebin

            self.filetypes = [fname.split("/")[-1].split(".")[-1] for fname in self.data_files]
            

            if resolution is None:
                self.resolution = 20. # COS
            else:
                self.resolution = resolution

            if name is None:
                if self.filetypes[0]=="fits":
                    self.name = self.data_files[0].split("hst_cos_")[-1].split("_")[0]
                else:
                    self.name = self.data_files[0].split("/")[-2]
            else:
                self.name = name

            if lines is None:
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

            if velspan is None:
                self.velspan = 500.
            else:
                self.velspan = velspan

            # if rebin_n is None:
            #     self.rebin_n = 1
            # else:
            #     self.rebin_n = rebin_n

            if pre_rebin:
                self.rebin_n = 1

            if rebin_method is None:
                self.rebin_method = "mean"
            else:
                self.rebin_method = rebin_method

            self.auto_resolution = auto_resolution

            customSimbad = Simbad()
            customSimbad.add_votable_fields("rvz_radvel", "rvz_type")

            if query_name is None:
                try:
                    self.source_info = customSimbad.query_object(self.name)
                except:
                    self.source_info = Simbad.query_object(self.name)
            else:
                try:
                    self.source_info = customSimbad.query_object(query_name)
                except:
                    self.source_info = Simbad.query_object(query_name)

            if self.source_info is None: #if simbad failed
                fits_file_inds = self.filetypes == "fits"
                from astropy.io import fits
                if np.sum(fits_file_inds) == 0:
                    with fits.open(self.data_files[fits_file_inds]) as fits_file:
                        self.source_info = {"RA":[], "DEC":[]}
                        self.source_info["RA"]=[float(fits_file[0].header["TARG_RA"]) * u.deg]
                        self.source_info["RA"][0] = self.source_info["RA"][0].to(u.hourangle).value
                        self.source_info["DEC"]=[float(fits_file[0].header["TARG_DEC"])]
                else:
                    with fits.open(self.data_files[fits_file_inds][0]) as fits_file:
                        self.source_info = {"RA":[], "DEC":[]}
                        self.source_info["RA"]=[float(fits_file[0].header["TARG_RA"]) * u.deg]
                        self.source_info["RA"][0] = self.source_info["RA"][0].to(u.hourangle).value
                        self.source_info["DEC"]=[float(fits_file[0].header["TARG_DEC"])]

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

            if redshift is None:
                self.redshift = self.redshift_from_rv
            else:
                self.redshift = redshift

            # open Dataset
            self.dataset = VoigtFit.DataSet(self.redshift)
            self.dataset.set_name(self.name)

            self.file_suffix = []
            for fn in self.data_files:
                if "g130m" in fn.lower():
                    if "G130M-N-DK" in fn:
                        self.file_suffix.append("G130M-N-DK")
                    elif "G130M-N" in fn:
                        self.file_suffix.append("G130M-N")
                    else:
                        self.file_suffix.append("G130M")
                elif "g160m" in fn.lower():
                    self.file_suffix.append("G160M")
                elif "e140m" in fn.lower():
                    self.file_suffix.append("E140M")
                elif "e140h" in fn.lower():
                    self.file_suffix.append("E140H")
                elif "LIF1" in fn.lower():
                    self.file_suffix.append("LIF1")
                else:
                    self.file_suffix.append(fn.split("{}_".format(self.name))[-1].split("_")[0].upper())

            self.file_suffix = np.array(self.file_suffix)

            
            # self.file_suffix = np.array([f.split("{}_".format(self.name))[-1].split("_")[0].upper() if ft == "fits" 
            #                              else f.split("_spec-")[-1] for f,ft in zip(self.data_files, self.filetypes)])
            if filter_regions:
            
                
                self.filter_dict = {}
                if use_DK_N:

                    try:
                        self.filter_dict["G160M"] = np.where(self.file_suffix == "G160M")[0][0]
                    except IndexError:
                        pass

                    try:
                        self.filter_dict["G130M"] = np.where(self.file_suffix == "G130M")[0][0]
                    except IndexError:
                        pass

                    try:
                        self.filter_dict["G130M-N-DK"] = np.where(self.file_suffix == "G130M-N-DK")[0][0]
                    except IndexError:
                        pass
                    
                    try:
                        self.filter_dict["LIF1"] = np.where(self.file_suffix == "LIF1")[0][0]
                    except IndexError:
                        pass

                    self.tag_file_pairs = {"OI_1302":["G130M-N-DK"], 
                                      "OI_1039":["LIF1"], 
                                      "SiII_1304":["G130M-N-DK"],
                                      "SiIV_1402":["G130M"],
                                      "SiIV_1393":["G130M"],
                                      "SiIII_1206":["G130M"],
                                      "SiII_1260":["G130M"],
                                      "SiII_1193":["G130M"],
                                      "SiII_1190":["G130M"],
                                      "CII_1334":["G130M"],
                                      "CIIa_1335.7":["G130M"],
                                      "CIIa_1335.71":["G130M"],
                                      "FeII_1144":["G130M"],
                                      "SII_1250":["G130M"],
                                      "SII_1253":["G130M"],
                                      "SII_1259":["G130M"],
                                      "NI_1200.7":["G130M"],
                                      "NI_1200":["G130M"],
                                      "NI_1199":["G130M"]}

                else:

                    try:
                        self.filter_dict["G160M"] = np.where(self.file_suffix == "G160M")[0][0]
                    except IndexError:
                        pass
                        
                    try:
                        self.filter_dict["G130M"] = np.where(self.file_suffix == "G130M")[0][0]
                    except IndexError:
                        pass

                    try:
                        self.filter_dict["G130M-N"] = np.where(self.file_suffix == "G130M-N")[0][0]
                    except IndexError:
                        pass
                    
                    try:
                        self.filter_dict["LIF1"] = np.where(self.file_suffix == "LIF1")[0][0]
                    except IndexError:
                        pass 

                    self.tag_file_pairs = {"OI_1302":["G130M-N", "G130M"],  
                                      "OI_1039":["LIF1"], 
                                      "SiII_1304":["G130M-N", "G130M"],
                                      "SiIV_1402":["G130M"],
                                      "SiIV_1393":["G130M"],
                                      "SiIII_1206":["G130M"],
                                      "SiII_1260":["G130M"],
                                      "SiII_1193":["G130M"],
                                      "SiII_1190":["G130M"],
                                      "CII_1334":["G130M"],
                                      "CIIa_1335.7":["G130M"],
                                      "CIIa_1335.71":["G130M"],
                                      "FeII_1144":["G130M"],
                                      "SII_1250":["G130M"],
                                      "SII_1253":["G130M"],
                                      "SII_1259":["G130M"],
                                      "NI_1200.7":["G130M"],
                                      "NI_1200":["G130M"],
                                      "NI_1199":["G130M"]}



            # read in data from text file
            for suffix,file,filetype in zip(self.file_suffix, self.data_files, self.filetypes):
                print("Loading data from file, {0}, with filter {1}".format(file.split("/")[-1], suffix))
                if auto_resolution:
                    if suffix == "G160M":
                        self.resolution = 15.
                        print("Setting G160M resolution to 15 km/s")
                    elif "e140m" in file:
                        self.resolution = 6.5
                    elif "e140h" in file:
                        self.resolution = 2.5
                    elif "g130m" in file:
                        self.resolution = 20.
                    elif "fuse" in file:
                        self.resolution = 20.
                    elif suffix == "LIF1":
                        self.resolution = 20.
                    else:
                        self.resolution = 15.

                if filetype == "fits":
                    tmp_table = Table.read(file, format = filetype)
                    wav = tmp_table["WAVELENGTH"].T.data
                    flux = tmp_table["FLUX"].T.data
                    err = tmp_table["ERROR"].T.data

                else:
                    try:
                        wav, flux, err, _,_, _,_, _,_ = np.loadtxt(file, unpack = True)
                    except ValueError:
                        wav, flux, err = np.loadtxt(file, unpack = True)

                mask = flux < 0
                mask |= np.isnan(flux)
                mask |= np.isinf(flux)
                if (suffix == "G130M-N-DK") & (manual_night_only_error != None):
                    err = manual_night_only_error * flux
                if ((suffix == "G130M-N")|(suffix == "G130M-N-DK")) & (shift_night_only_at_OI != None):
                    l0_ref = 1302.1680
                    l_ref = l0_ref*(self.redshift+1)
                    wav_shift = shift_night_only_at_OI / speed_of_light.to(u.km/u.s).value * l_ref
                    wav += wav_shift
                    print("shifting Night Only data by {}".format(wav_shift))
                if (suffix == "G160M") & (shift_g160_at_1526 != None):
                    l0_ref = 1526.7066
                    l_ref = l0_ref*(self.redshift+1)
                    wav_shift = shift_g160_at_1526 / speed_of_light.to(u.km/u.s).value * l_ref
                    wav += wav_shift
                    print("shifting G160M data by {}".format(wav_shift))
                if self.pre_rebin:
                    if rebin_n is None:
                        if suffix == "G160M":
                            self.rebin_n = 3
                        elif suffix in ["G130M", "LIF1"]:
                            self.rebin_n = 5
                        # elif "fuse" in file:
                        #     self.rebin_n = 5
                        elif suffix == "E140M":
                            self.rebin_n = 1
                        elif suffix == "E140H":
                            self.rebin_n = 1
                        else:
                            self.rebin_n = 1
                    wl_r, spec_r, err_r = rebin_spectrum(wav[~mask], flux[~mask], err[~mask], 
                                                         self.rebin_n, method = self.rebin_method)
                    print(self.rebin_n)
                    self.dataset.add_data(wl_r, spec_r, self.resolution, 
                                      err = err_r, 
                                      normalized = False, filename = suffix)
                else:
                    self.dataset.add_data(wav[~mask], flux[~mask], self.resolution, 
                                      err = err[~mask], 
                                      normalized = False, filename = suffix)


            # Add relevent lines to dataset
            for line in self.lines:
                print(line)
                try:
                    self.dataset.add_line(line, velspan = self.velspan)
                except ValueError:
                    print("skipping {line} - Value Error Encountered.")

            self.specID_list = [region.specID for region in self.dataset.regions]
            self.specID_to_suffix = {}
            for specID,suffix in zip(self.specID_list, self.file_suffix):
                self.specID_to_suffix["{}".format(specID)] = suffix
            if filter_regions:
                self.filter_regions()

        else:

            self.pre_rebin = pre_rebin
            #loading from existing dataset
            self.data_files = [filename]

            if resolution is None:
                self.resolution = 20.
            else:
                self.resolution = resolution

            # load dataset
            self.dataset = VoigtFit.load_dataset(filename)

            if name is None:
                self.name = self.dataset.name
            else:
                self.name = name

            regions = self.dataset.regions
            self.lines = []
            for region in regions:
                self.lines = np.concatenate([self.lines, [line.tag for line in region.lines]])

            self.velspan = self.dataset.velspan

            if rebin_n is None:
                self.rebin_n = 1
            else:
                self.rebin_n = rebin_n

            if self.pre_rebin:
                self.rebin_n = 1
                rebin_n = 1

            if rebin_method is None:
                self.rebin_method = "mean"
            else:
                self.rebin_method = rebin_method

            if spec_file is None:
                customSimbad = Simbad()
                # customSimbad.add_votable_fields("rvz_radvel", "rvz_type")

                # print(f"getting Simbad Query for {self.name}")
                if query_name is None:
                    self.source_info = customSimbad.query_object(self.name)
                else:
                    self.source_info = customSimbad.query_object(query_name)

            else:
                from astropy.io import fits
                with fits.open(spec_file) as fits_file:
                    self.source_info = {"RA":[], "DEC":[]}
                    self.source_info["RA"]=[float(fits_file[0].header["TARG_RA"]) * u.deg]
                    self.source_info["RA"][0] = self.source_info["RA"][0].to(u.hourangle).value
                    self.source_info["DEC"]=[float(fits_file[0].header["TARG_DEC"])]


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

            self.specID_list = [region.specID for region in self.dataset.regions]
            # self.specID_to_suffix = {}
            # for specID,suffix in zip(self.specID_list, self.file_suffix):
            #     specID_to_suffix["{}".format(specID)] = suffix

    def save_dataset(self, filename, in_same_folder = False):
        data.dataset.data_filenames = np.array(data.dataset.data_filenames, dtype= "S")
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
    
    def __init__(self, source_name, source_coord = None, 
                 source_info = None, 
                 distance_grid_command = None, 
                 neutral_column_density_grid = None, 
                 stop_OI_column = None,
                 spectra_template_filename = None, 
                 egb = None, 
                 egb_redshift = None, 
                 cosmic_rays_background = True,
                 species = None,
                 metalicity_grid_command = None,
                 hden_grid_command = None,):

        self.source_name = source_name

        if source_info is None:
            self.source_info = Simbad.query_object(self.source_name)
        else:
            self.source_info = source_info

        if source_coord is None:
            self.source_coord = SkyCoord(ra = self.source_info["RA"], 
                    dec = self.source_info["DEC"], 
                    unit = (u.hourangle, u.deg), 
                    frame = "icrs").transform_to('galactic')
        else:
            self.source_coord = source_coord.transform_to('galactic')

        if distance_grid_command is None:
            self.distance_grid_command = [22.9,23.55,0.05]
        else:
            self.distance_grid_command = distance_grid_command

        self.distance_grid = 10**np.arange(*self.distance_grid_command) * u.cm
        self.distnace_grid = self.distance_grid.to(u.kpc)


        self.neutral_column_density_grid = neutral_column_density_grid

        self.stop_OI_column = stop_OI_column

        if spectra_template_filename != None:
            self.spectra_template_filename = spectra_template_filename
        else:
            self.spectra_template_filename = os.path.join(directory,"data/JBH_RadiationField/Fox+2005_MW.sed")


        if egb is None:
            self.egb = "KS18"
        else:
            self.egb = egb

        if egb_redshift is None:
            self.egb_redshift = 0.
        else:
            self.egb_redshift = egb_redshift

        self.cosmic_rays_background = cosmic_rays_background

        if species is None:
            self.species = ["H", "H+", "Si+", "Si+2", "Si+3", "C+", "C+3", "Fe+", "Al+", "O"]
        else:
            self.species = species


        if metalicity_grid_command is None:
            self.metalicity_grid_command = [-0.7,-0.1,0.2]
        else:
            self.metalicity_grid_command = metalicity

        if hden_grid_command is None:
            self.hden_grid_command = [-3,0,0.5]
        else:
            self.hden_grid_command = hden_grid_command

        self.hden_grid = 10**np.arange(*self.hden_grid_command) * u.cm**-3

        self.input_filename = None


        


    def get_input_file(self, stop_neutral_column_density = None, distance = None, save = False, 
                        grid_colden = False, grid_metals = True, stop_OI_column = None):
        """
        Returns string of input file as list for each line of file
        """ 

        if distance is None:
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

        if stop_OI_column is None:
            stop_OI_column = self.stop_OI_column



        file_lines = []


        file_lines.append(f'title {self.source_name}')
        file_lines.append('# Input Spectrum File')
        file_lines.append(f'table SED "{input_spectra_filename}"')
        file_lines.append('# Normalization')
        file_lines.append(f'phi(H) = {np.log10(norms["TOTAL"].value[0])}')
        file_lines.append('# Extragalactic Background')
        file_lines.append(f'Table {self.egb} redshift {self.egb_redshift}')
        file_lines.append('# hden')
        file_lines.append('hden -3.0 vary')
        file_lines.append('grid {} {} {}'.format(*self.hden_grid_command))

        if self.cosmic_rays_background:
            file_lines.append('cosmic rays background')

        file_lines.append('constant density')
        file_lines.append('# Metalcity')
        if grid_metals:
            file_lines.append('metals -0.3 log vary')
            file_lines.append('grid {} {} {}'.format(*self.metalicity_grid_command))
        else:
            file_lines.append(f'metals {self.metalicity_grid_command} log')
        file_lines.append('# Stop condition')
        if stop_OI_column is None:
            file_lines.append(f'stop neutral column density {stop_neutral_column_density}')
            stop_val = stop_neutral_column_density
        else:
            file_lines.append(f'stop column density "O" {stop_OI_column}')
            stop_val = stop_OI_column
        

        file_lines.append('double optical depths')
        file_lines.append('iterate to convergence')
        file_lines.append('save grid separate "distance_kpc_{0:.1f}_stop_{1:.2f}_gridrun.grd"'.format(distance.value, stop_val))
        file_lines.append('save species column densities last separate "distance_kpc_{0:.1f}_stop_{1:.2f}_colden.col" no hash'.format(distance.value, stop_val))
        for species in self.species:
            file_lines.append(f'"{species}"')

        file_lines.append('end of species')
        file_lines.append('print last')
        file_lines.append('plot continuum')

        if save:

            with open('distance_kpc_{0:.1f}_stop_{1:.2f}_input.in'.format(distance.value, stop_val), 'w') as f:
                for line in file_lines:
                    print(line, file = f)

            self.input_filename = 'distance_kpc_{0:.1f}_stop_{1:.2f}_input.in'.format(distance.value, stop_val)

        return file_lines

    def print_input_file(self, file_lines):
        for line in file_lines:
            print(line)




    def grid_viewer(self, data, meas = None,
                    figsize = None, 
                    cloudy_results = None,
                    ions = None, 
                    cmap = None, 
                    vel_range = None,
                    ):
        return CloudyGridViewer(cloudy = self, data = data, 
                                figsize = figsize, 
                                cloudy_results = cloudy_results, 
                                ions = ions, 
                                cmap = cmap, 
                                vel_range = vel_range, 
                                meas = meas)






class CloudyGridViewer(object):
    
    def __init__(self, cloudy = None,
                       data = None, 
                       figsize = None, 
                       cloudy_results = None, 
                       ions = None,  
                       cmap = None, 
                 vel_range = None,
                 meas = None):
        """
        Interactive plot of cloudy results with slides to control the plotted distances, stop_colN, or METALS
        """

        if ions is None:
            self.ions = ["HI", "HII", "OI", "FeII", "AlII", "SiII", "SiIII", "SiIV", "CII", "CIV"]
        else:
            self.ions = ions

        if figsize is None:
            figsize = (8.5,11)

        if cloudy_results is None:
            assert cloudy.cloudy_results != None
            self.cloudy_results = cloudy.cloudy_results
        else:
            self.cloudy_results = cloudy_results

        if cmap is None:
            self.cmap = "plasma"
        else:
            self.cmap = cmap
            
        if vel_range is None:
            self.vel_min = -200
            self.vel_max = 600
        else:
            self.vel_min = vel_range[0]
            self.vel_max = vel_range[1]
            
        if meas is None:
            self.meas = cloudy.meas  
        else:
            self.meas = meas  
        self.data = data
        self.cloudy = cloudy


        # set default values
        self.distance_0 = np.sort(self.cloudy_results["DISTANCE"])[(int(len(self.cloudy_results)/2))]
        self.stop_coln_0 = np.sort(self.cloudy_results["STOP_COLN"])[(int(len(self.cloudy_results)/2))]
        self.metals_0 = np.sort(self.cloudy_results["METALS"])[(int(len(self.cloudy_results)/2))]

        self.delta_distance = np.diff(np.unique(self.cloudy_results["DISTANCE"]))[0]
        self.delta_stop_coln = np.diff(np.unique(self.cloudy_results["STOP_COLN"]))[0]
        self.delta_metals = np.diff(np.unique(self.cloudy_results["METALS"]))[0]



        self.fig, self.axs = plt.subplots(5,2, figsize = figsize)

        plt.subplots_adjust(left = 0.25, top = 0.94, bottom = 0.2, right = .94,
                            hspace = 0.5, wspace = .12)

        

        self.grid_data = {"DISTANCE":np.unique(self.cloudy_results["DISTANCE"]),
                     "STOP_COLN":np.unique(self.cloudy_results["STOP_COLN"]),
                     "METALS":np.unique(self.cloudy_results["METALS"])}
        
        self.as_cmap = "DISTANCE"
        self.norm = {"DISTANCE":Normalize(vmin = np.min(self.grid_data["DISTANCE"])-5., 
                                          vmax = np.max(self.grid_data["DISTANCE"])+5.),
                "STOP_COLN":Normalize(vmin = np.min(self.grid_data["STOP_COLN"])-.5, 
                                          vmax = np.max(self.grid_data["STOP_COLN"])+.5),
                "METALS":Normalize(vmin = np.min(self.grid_data["METALS"])-.2, 
                                          vmax = np.max(self.grid_data["METALS"])+.2)}

        self.sm = cm.ScalarMappable(cmap = self.cmap, norm = self.norm[self.as_cmap])

        self.xlim = (np.min(self.cloudy_results["HDEN"]), np.max(self.cloudy_results["HDEN"]))

        self.max_lines = np.max([len(self.grid_data["DISTANCE"]),
                            len(self.grid_data["STOP_COLN"]),
                            len(self.grid_data["METALS"])])

        #initial lines
        self.lines_list = []
        for ax,ion in zip(self.axs.flatten(), self.ions):
            lines = []
            nlines = 0
            for val in self.grid_data[self.as_cmap]:
                nlines +=1
                color = self.sm.to_rgba(val)
                mask = self.cloudy_results[self.as_cmap] == val
                if self.as_cmap == "DISTANCE":
                    mask &= self.cloudy_results["STOP_COLN"] == self.stop_coln_0
                    mask &= self.cloudy_results["METALS"] == self.metals_0
                elif self.as_cmap == "STOP_COLN":
                    mask &= self.cloudy_results["DISTANCE"] == self.distance_0
                    mask &= self.cloudy_results["METALS"] == self.metals_0
                else:
                    mask &= self.cloudy_results["DISTANCE"] == self.distance_0
                    mask &= self.cloudy_results["STOP_COLN"] == self.stop_coln_0

                yy = np.ma.masked_array(data = self.cloudy_results[f"N_{ion}"][mask].value, 
                                        mask = self.cloudy_results[f"N_{ion}"][mask].value <= 0.)
                yy = np.ma.log10(yy)
                l, = ax.plot(self.cloudy_results["HDEN"][mask], 
                                yy, 
                                color = self.sm.to_rgba(val), lw = 2, alpha = 0.8,
                             label = val)
                ax.set_xlim(self.xlim)
                ax.set_title(ion, fontsize = 12)
                lines.append(l)

            while nlines < self.max_lines:
                l, = ax.plot(self.cloudy_results["HDEN"][mask], yy, lw = 2, alpha = 0.0)
                lines.append(l)
                nlines+=1

            self.lines_list.append(lines)
        
        handles, labels = self.axs.flatten()[0].get_legend_handles_labels()
        self.lg = self.fig.legend(handles, labels, loc='upper center', ncol = 8)
        
        self.title = self.fig.suptitle(self.cloudy.source_name, x = 0.1, y = 0.83, fontweight = "bold")
        
        self.axs[-1][0].set_xlabel(r"$\log_{10}(n_H)$", fontsize = 12)
        self.axs[-1][1].set_xlabel(r"$\log_{10}(n_H)$", fontsize = 12)
        for ax in self.axs[:,0]:
            ax.set_ylabel(r"$\log_{10}(N)$", fontsize = 12)
        for ax in self.axs[:,1]:
            ax.yaxis.tick_right()

        def plot_result(as_cmap, distance = self.distance_0, stop_coln = self.stop_coln_0, metals = self.metals_0):
            for lines,ion in zip(self.lines_list, self.ions):

                for val,l in zip(self.grid_data[self.as_cmap],lines[:len(self.grid_data[self.as_cmap])]):
                    color = self.sm.to_rgba(val)
                    mask = self.cloudy_results[self.as_cmap] == val
                    if as_cmap == "DISTANCE":
                        mask &= self.cloudy_results["STOP_COLN"] < stop_coln + self.delta_stop_coln/2
                        mask &= self.cloudy_results["STOP_COLN"] > stop_coln - self.delta_stop_coln/2
                        mask &= self.cloudy_results["METALS"] < metals + self.delta_metals/2
                        mask &= self.cloudy_results["METALS"] > metals - self.delta_metals/2
                    elif as_cmap == "STOP_COLN":
                        mask &= self.cloudy_results["DISTANCE"] < distance + self.delta_distance/2
                        mask &= self.cloudy_results["DISTANCE"] > distance - self.delta_distance/2
                        mask &= self.cloudy_results["METALS"] < metals + self.delta_metals/2
                        mask &= self.cloudy_results["METALS"] > metals - self.delta_metals/2
                    else:
                        mask &= self.cloudy_results["DISTANCE"] < distance + self.delta_distance/2
                        mask &= self.cloudy_results["DISTANCE"] > distance - self.delta_distance/2
                        mask &= self.cloudy_results["STOP_COLN"] < stop_coln + self.delta_stop_coln/2
                        mask &= self.cloudy_results["STOP_COLN"] > stop_coln - self.delta_stop_coln/2

                    yy = np.ma.masked_array(data = self.cloudy_results[f"N_{ion}"][mask].value, 
                                        mask = self.cloudy_results[f"N_{ion}"][mask].value <= 0.)
                    yy = np.ma.log10(yy)
                    self.yy = yy
                    l.set_ydata(yy)
                    l.set_alpha(0.8)
                    l.set_color(color)
                    l.set_label(val)

                for l in lines[len(self.grid_data[self.as_cmap]):]:
                    l.set_alpha(0.0)
                    l.set_label(None)
                    
                    
            for ax in self.axs.flatten():
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(self.xlim)
            self.lg.remove()
            handles, labels = self.axs.flatten()[0].get_legend_handles_labels()
            self.lg = self.fig.legend(handles, labels, loc='upper center', ncol = 8)


        self.axcolor = 'lightgoldenrodyellow'
        self.axdist = plt.axes([0.02, 0.2, 0.02, 0.57], facecolor=self.axcolor)
        self.axcoln = plt.axes([0.08, 0.2, 0.02, 0.57], facecolor=self.axcolor)
        self.axmetal = plt.axes([0.14, 0.2, 0.02, 0.57], facecolor=self.axcolor)

        self.sdist = Slider(self.axdist, 'D', 
                       np.min(self.grid_data["DISTANCE"]), 
                       np.max(self.grid_data["DISTANCE"]), 
                       valinit=self.distance_0, 
                       valstep=self.delta_distance, 
                       orientation = "vertical")

        self.smetal = Slider(self.axmetal, 'Z', 
                       np.min(self.grid_data["METALS"]), 
                       np.max(self.grid_data["METALS"]), 
                       valinit=self.metals_0, 
                       valstep=self.delta_metals, 
                       orientation = "vertical")

        self.scoln = Slider(self.axcoln, 'N', 
                       np.min(self.grid_data["STOP_COLN"]), 
                       np.max(self.grid_data["STOP_COLN"]), 
                       valinit=self.stop_coln_0, 
                       valstep=self.delta_stop_coln, 
                       orientation = "vertical")

        self.rax = plt.axes([0.015, 0.85, 0.14, 0.1], facecolor=self.axcolor)
        self.radio = RadioButtons(self.rax, ('DISTANCE', 'STOP_COLN', 'METALS'), active=0)

        
        
        self.axvw = plt.axes([.18, .02, .02, .13], facecolor = self.axcolor)
        self.svw = Slider(self.axvw, r"$\Delta v$", 
                          5., 40., valinit = 15., orientation = "vertical", valstep = 0.25)
        
        self.rabs_ax = plt.axes([0.02,0.02, .15, .13], facecolor = self.axcolor)
        self.radio_abs = RadioButtons(self.rabs_ax, (r"OI $\lambda$1302", 
                                                     r"AlII $\lambda$1670",
                                                     r"SiII $\lambda$1190", 
                                                     r"SiIII $\lambda$1206",
                                                     r"SiIV $\lambda$1393",
                                                     r"CIV $\lambda$1548"), active = 0)
        
        self.axvcen = plt.axes([.25, .01, .69, .015], facecolor = self.axcolor)
        self.svcen = Slider(self.axvcen, r"$v$",
                            self.vel_min, self.vel_max, valinit = 250., valstep = 1.)
        
        self.axabs = plt.axes([.25, .05, .69, .1])
        self.axabs.set_xlim(self.vel_min,self.vel_max)
        self.axabs.yaxis.set_label_position("right")
        
        #init spectra
        self.line_to_tag = {r"OI $\lambda$1302":"OI_1302", 
                            r"AlII $\lambda$1670":"AlII_1670",
                            r"SiII $\lambda$1190":"SiII_1190",
                            r"SiIII $\lambda$1206":"SiIII_1206",
                            r"SiIV $\lambda$1393":"SiIV_1393",
                            r"CIV $\lambda$1548":"CIV_1548"}
                
        
        def plot_spec(line_label = self.radio_abs.value_selected):
            # Find the right region
            self.axabs.clear()
            line_label = self.radio_abs.value_selected
            for ell,region in enumerate(self.data.voigtfit[self.cloudy.source_name]["LOW"].dataset.regions):
                for sub_ind,line in enumerate(region.lines):
                    if self.line_to_tag[line_label] == line.tag:
                        self.data.voigtfit[self.cloudy.source_name]["LOW"].plot_region_fit(ell, 
                                                                                    sub_region_ind = sub_ind, 
                                                                                    vel_range = [self.vel_min, 
                                                                                                 self.vel_max],
                                                                                    ax = self.axabs, 
                                                                                    labelx = False, 
                                                                                    labely = False, 
                                                                                    lw = 1, 
                                                                                    alpha = 0.8, 
                                                                                    fit_kwargs = {"lw":1, 
                                                                                                  "alpha":0.8},
                                                                                    comp_kwargs = {"lw":2},
                                                                                    comp_scale = 0.2, 
                                                                                    plot_indiv_comps = True,
                                                                                           ylabel_as_ion = True)
            for ell,region in enumerate(self.data.voigtfit[self.cloudy.source_name]["HIGH"].dataset.regions):
                for sub_ind,line in enumerate(region.lines):
                    if self.line_to_tag[line_label] == line.tag:
                        self.data.voigtfit[self.cloudy.source_name]["HIGH"].plot_region_fit(ell, 
                                                                                    sub_region_ind = sub_ind, 
                                                                                    vel_range = [self.vel_min, 
                                                                                                 self.vel_max],
                                                                                    ax = self.axabs, 
                                                                                    labelx = False, 
                                                                                    labely = False, 
                                                                                    lw = 1, 
                                                                                    alpha = 0.8, 
                                                                                    fit_kwargs = {"lw":1, 
                                                                                                  "alpha":0.8},
                                                                                    comp_kwargs = {"lw":2},
                                                                                    comp_scale = 0.3, 
                                                                                    plot_indiv_comps = True, 
                                                                                            ylabel_as_ion = True)
                        
            self.axabs.set_ylabel("Normalized\nFlux", fontsize = 12)
            # add vel markers
            ylim = self.axabs.get_ylim()
            self.axabs.plot([self.svcen.val, self.svcen.val], ylim, color = "k", ls = ":", 
                            lw = 2, alpha = 0.5, zorder = -1)
            self.axabs.fill_between([self.svcen.val - self.svw.val, self.svcen.val + self.svw.val], 
                                    [ylim[0], ylim[0]], 
                                    [ylim[1], ylim[1]], color = "k", alpha = 0.04, zorder = -1)
            
        
                        
        plot_spec(self.radio_abs.value_selected)
        
        
        
        #init lines
        self.meas_lines = []
        for ion,ax in zip(self.ions, self.axs.flatten()):
            l0, = ax.plot(self.xlim,[12,12],lw = 2, color = "k", ls = "--", alpha = 0., zorder = -1)
            
            l1, = ax.plot(self.xlim,[12,12],lw = 2, color = "k", ls = ":", alpha = 0., zorder = -1)
            
            l2, = ax.plot(self.xlim,[12,12],lw = 2, color = "k", ls = ":", alpha = 0., zorder = -1)
            
            self.meas_lines.append([l0,l1,l2])
        
        def add_obs_lines():
            for ion,lines in zip(self.ions, self.meas_lines):
                #check for matching component
                vel_mask = self.meas["V"].value <= (self.svcen.val + self.svw.val)
                vel_mask &= self.meas["V"].value >= (self.svcen.val - self.svw.val)
                ion_mask = [comp.split("_")[-1] == ion for comp in self.meas["COMP"]]
                mask = vel_mask & ion_mask
                if np.sum(mask)>0:
                    res_N = np.log10(self.meas["N"].value)[mask]
                    res_err_N = np.log10(self.meas["ERR_N"].value)[mask]

                    res_v = self.meas["V"].value[mask]

                    # check for multiple
                    if len(res_v) > 1:
                        use_ind = np.argmin(np.abs(res_v - self.svcen.val))
                        res_v = res_v[use_ind]
                        res_N = res_N[use_ind]
                        res_err_N = res_err_N[use_ind]
                    if np.isinf(res_err_N):
                        res_err_N = 0.
                    
                    lines[0].set_ydata([res_N, res_N])
                    lines[1].set_ydata([res_N-res_err_N, res_N-res_err_N])
                    lines[2].set_ydata([res_N+res_err_N, res_N+res_err_N])
                    
                    lines[0].set_alpha(0.7)
                    lines[1].set_alpha(0.7)
                    lines[2].set_alpha(0.7)
                else:
                    lines[0].set_alpha(0.)
                    lines[1].set_alpha(0.)
                    lines[2].set_alpha(0.)
                    
        def update_cmap_parameter(label):
            self.as_cmap = label
            self.sm = cm.ScalarMappable(cmap = self.cmap, norm = self.norm[self.as_cmap])
            plot_result(self.as_cmap, distance = self.sdist.val, stop_coln = self.scoln.val, metals = self.smetal.val)
            add_obs_lines()
            plot_spec()
            self.fig.canvas.draw_idle()
        self.radio.on_clicked(update_cmap_parameter)

        def update(val):
            plot_result(self.as_cmap, distance = self.sdist.val, stop_coln = self.scoln.val, metals = self.smetal.val)
            add_obs_lines()
            plot_spec()
            self.fig.canvas.draw_idle()
        self.sdist.on_changed(update)
        self.smetal.on_changed(update)
        self.scoln.on_changed(update)
        
        self.radio_abs.on_clicked(update)
        self.svw.on_changed(update)
        self.svcen.on_changed(update)
                
    

















