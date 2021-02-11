import logging
import pickle

#pytorch
import torch
#sbi
import sbi.utils as utils
from sbi.inference.base import infer 

from .sbiKitMixin import sbiKitMixin
import numpy as np



class sbiKit(sbiKitMixin):
    """
    Simulation Based Inference tool kit for convenience

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

    def __init__(self, UVSpectra, ion, mask_limits = True, from_file = None):
        if from_file.__class__ is str:
            with open(from_file, "rb") as f:
                c = pickle.load(f)
                c.from_file = from_file
                return c


        if ion.__class__ is str:
            z = UVSpectra["N_{}".format(ion)].copy()
            mask = np.isnan(z)
            mask |= np.isinf(z)
            if mask_limits:
                mask |= UVSpectra["N_{}_UPPERLIMIT".format(ion)]
                mask |= UVSpectra["N_{}_LOWERLIMIT".format(ion)]
        else:
            z = np.zeros_like(UVSpectra["N_{}".format(ion[0])])
            mask = np.zeros_like(z, dtype = bool)
            for ion_name in ion:
                z += UVSpectra["N_{}".format(ion_name)].copy()
                mask |= np.isinf(z)
                mask |= np.isnan(z)
                if mask_limits:
                    mask |= UVSpectra["N_{}_UPPERLIMIT".format(ion_name)]
                    mask |= UVSpectra["N_{}_LOWERLIMIT".format(ion_name)]

        # get impact parameters
        b = UVSpectra["LMC_B"].copy()

        # setup
        self.observations = torch.tensor(z[~mask])
        self.b = torch.tensor(b[~mask])
        self.n_obs = len(self.observations)
        self.ion = ion
        self.mask_limits = mask_limits
        self.from_file = from_file
        self.ready = False

        

    def get_posterior(self, prior_kwargs = {}, sim_kwargs = {}, setup = True, **kwargs):
        """
        returns posterior ( or creates one)

        Parameters:
        prior_kwargs: `dict`
            keywords passed to flat_prior
        sim_kwargs: `dict`
            keywords passed to exp_simulator
        kwargs:
            passed on to sbi.inference.base.infer
        """

        if (len(prior_kwargs) + len(sim_kwargs) + len(kwargs)) == 0:
            # no extra kwargs provided, return existing or use default
            if hasattr(self, "posterior"):
                return self.posterior
            else:
                if self.ready:
                    return self.train_posterior()
                else:
                    self.prep_for_inference()
                    return self.train_posterior()

        elif (len(prior_kwargs) + len(sim_kwargs)) == 0:
            if self.ready:
                return self.train_posterior(**kwargs)
            else:
                self.prep_for_inference()
                return self.train_posterior(**kwargs)

        else:
            self.prep_for_inference(prior_kwargs = prior_kwargs, sim_kwargs = sim_kwargs)
            return self.train_posterior(**kwargs)

    def get_log_prob(self, num_samples = None, **kwargs):
        """
        Returns log probability using posterior

        Parameters
        ----------
        num_samples:   `int`, optional, must be keyword
            number of samples to use from observations
        kwargs:
            keywords passed to posterior.log_prob

        Returns
        -------
        samples: samples from posterior
        log_prob: log_prob for samples
        """

        if num_samples == None:
            num_samples = 10000
        samples = self.posterior.sample((num_samples,), x = self.observations)
        return samples, self.posterior.log_prob(samples, x = self.observations, **kwargs)








