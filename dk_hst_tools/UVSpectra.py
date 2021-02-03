import logging

import numpy as np 
from astropy import units as u 
from astropy.table import Table 
from astropy.coordinates import SkyCoord

from .uvDataMixin import UVSpectraMixin

import os

import glob
import io
import pandas as pd

directory = os.path.dirname(__file__)

class UVSpectra(UVSpectraMixin, Table):
    """
    Core UV Spectra class

    Load, view, manipulate, and plot basic results from Bart's analysis files

    Parameters
    ----------

    directory: `str`, 
        Directory where data is stored, 
        organized as one folder per source/direction
    bart: `bool`, optional, must be keyword
        if True (default) assumes reading in Bart Wakker's data analysis products

    """

    def __init__(self, directory, bart = True, **kwargs):

        # Read in paths for data

        self.path = directory
        self.source_dirs = glob.glob(os.path.join(directory,"*"))

        # Get source names:
        self.source_names = [path.split("/")[-1] for path in self.source_dirs]

        self.abund_files = glob.glob(os.path.join(self.path,"*/*ABUND.txt"))
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

        frames = [read_abund_file(abund_file) for abund_file in self.abund_files]


        self.raw_df = pd.concat(frames)
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
            lower_limit_mask = [str(val)[0] == "<" for val in self.raw_table[species]]
            new_col = []
            for val in self.raw_table[species]:
                try:
                    new_col.append(float(str(val).split("<")[-1].split(">")[-1]))
                except ValueError:
                    new_col.append(np.nan)
            new_col = np.array(new_col)

            self.raw_table[species] = 10**new_col * u.cm**-2
            self.raw_table["{}_UPPERLIMIT".format(species)] = upper_limit_mask
            self.raw_table["{}_LOWERLIMIT".format(species)] = upper_limit_mask



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


        super().__init__(data = self.raw_table, **kwargs)

        
        