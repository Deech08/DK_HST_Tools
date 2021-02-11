import logging

import numpy as np 
from astropy import units as u 
from astropy.table import Table 
from astropy.coordinates import SkyCoord
from astropy.table.column import (BaseColumn, Column, MaskedColumn, _auto_names, FalseArray,
                     col_copy, _convert_sequence_data_to_array)

from .uvDataMixin import UVSpectraMixin

import os

import glob
import io
import pandas as pd

from astroquery.simbad import Simbad

directory = os.path.dirname(__file__)

cite_these = {
    "pymccorrelation":"https://github.com/privong/pymccorrelation",
    "Kendall's Tau with censoring": "https://ui.adsabs.harvard.edu/abs/1986ApJ...306..490I/abstract",
    "pyKrige":"DOI:10.5281/zenodo.3991907",
    "sbi_python":"DOI:10.21105/joss.02505",
    "Simulation Based Inference Review":"https://doi.org/10.1073/pnas.1912789117"
}

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
                 **kwargs):

        if bart:
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

        else:
            raise NotImplementedError("Not yet implemented to handle data not processed by Bart Wakker!")

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
                               SMC_coords = self.SMC_coords)
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
        