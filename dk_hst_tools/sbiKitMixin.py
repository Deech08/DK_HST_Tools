import logging
import numpy as np
import matplotlib.pyplot as plt
# pytorch
import torch
# sbi
import sbi.utils as utils
from sbi.inference.base import infer


class sbiKitMixin(object):
    """
    Mixin class for sbiKit
    """

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

    def exp_simulator(self, params, log = True, distance = 50.):
        """
        Makes simulator function

        Parameters
        ----------
        log: `bool`
            if provided, density parameter is in log_10 space
        distance, `number`, 
            distance to LMC in kpc
        """
        n_0, h_r = params
        if log:
            return self.exponential_halo_simulator(self.b, 
                                                   10**n_0,
                                                   h_r, 
                                                   distance = distance)
        else:
            return self.exponential_halo_simulator(self.b, 
                                                   n_0, 
                                                   h_r, 
                                                   distance = distance)

    def flat_prior(self, low = None, high = None, log = True):
        """
        BoxUniform prior
        """


        if low == None:
            if log:
                low = torch.tensor([-7, .5])
            else:
                low = torch.tensor([1e-7, 10**.5])
        if high == None:
            if log:
                high = torch.tensor([-2.5, 20])
            else:
                high = torch.tensor([10**-2.5, 20])


        return utils.BoxUniform(low = low, high = high)

    def prep_for_inference(self, prior_kwargs = {}, sim_kwargs = {}):
        """
        Prepares for inference

        Parameters
        ----------

        prior_kwargs: `dict`
            keywords passed to flat_prior
        sim_kwargs: `dict`
            keywords passed to exp_simulator
        """

        # check sim kwargs:
        def sim(params, **sim_kwargs):
            return self.exp_simulator(params, **sim_kwargs)

        self.simulator = sim

        # prior
        self.prior = self.flat_prior(**prior_kwargs)

        self.ready = True

    def train_posterior(self, **kwargs):
        """
        Trains posterior from simulator and prior

        Parameters
        ----------
        method: `str`, optional, must be keyword
            method to use in SBI, default to "SNPE"
        num_simulations: `int`, optional, must be keyword
            number of simulations to train with, defaults of 1000
        kwargs: 
            additional kwargs passed on to sbi.inference.base.infer

        Returns
        -------
        Posterior
        """

        # make sure ready:
        if not self.ready:
            raise ValueError("Not ready for training. make sure to run prep_for_inference!")


        from sbi.inference.base import infer

        if "method" not in kwargs:
            kwargs["method"] = "SNPE"

        if "num_simulations" not in kwargs:
            kwargs["num_simulations"] = 1000

        self.posterior = infer(self.simulator, self.prior, **kwargs) 

    def jointplot(self, samples = None, 
                  num_samples = None, 
                  label_meds = True,
                  lp_kwargs = {}, 
                  label_line_kwargs = {},
                  **kwargs):
        """
        Greates jointplot of samples

        samples:
            samples to plot, if not provided, gets them
        num_samples:
            number of samples to get if needed
        label_meds:
            if True, labels median values
        lp_kwargs: `dict`
            keywords passed on to self.get_log_prob
        label_line_kwargs: `dict`
            keywords passed to plotting of label lines
        """

        if samples == None:
            samples, _ = self.get_log_prob(num_samples = num_samples, **lp_kwargs)

        import seaborn as sns

        # Check kwargs:
        if "kind" not in kwargs:
            kwargs["kind"] = "kde"

        if kwargs["kind"] == "kde":
            if "fill" not in kwargs:
                kwargs["fill"] = True
                if "levels" not in kwargs:
                    kwargs["levels"] = 64
                if "marginal_kws" not in kwargs:
                    kwargs["marginal_kws"] = {"fill":True}



        g = sns.jointplot(x = samples[:,0], 
                          y = samples[:,1], 
                          **kwargs)

        x_pers = torch.quantile(samples[:,0], torch.tensor([.16, .5, .84]))
        y_pers = torch.quantile(samples[:,1], torch.tensor([.16, .5, .84]))

        xlim = g.ax_joint.get_xlim()
        ylim = g.ax_joint.get_ylim()

        if "color" not in label_line_kwargs:
            label_line_kwargs["color"] = "k"
        if "alpha" not in label_line_kwargs:
            label_line_kwargs["alpha"] = 0.7
        if "lw" not in label_line_kwargs:
            label_line_kwargs["lw"] = 2

        _ = g.ax_joint.plot(xlim, [y_pers[1], y_pers[1]], **label_line_kwargs, ls = "-")
        _ = g.ax_joint.plot(xlim, [y_pers[0], y_pers[0]], **label_line_kwargs, ls = ":")
        _ = g.ax_joint.plot(xlim, [y_pers[2], y_pers[2]], **label_line_kwargs, ls = ":")

        _ = g.ax_joint.plot([x_pers[1], x_pers[1]], ylim, **label_line_kwargs, ls = "-")
        _ = g.ax_joint.plot([x_pers[0], x_pers[0]], ylim, **label_line_kwargs, ls = ":")
        _ = g.ax_joint.plot([x_pers[2], x_pers[2]], ylim, **label_line_kwargs, ls = ":")

        # add text labels
        if torch.any(samples[:,0] < 0.):
            _ = g.ax_joint.set_xlabel(r"$\log(n_0/cm^{-3})$", fontsize = 12)
            xl = r"$\log(n_0/cm^{{-3}}) = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(x_pers[1], 
                                                                                   x_pers[2] - x_pers[1], 
                                                                                   x_pers[1] - x_pers[0])
        else:
            _ = g.ax_joint.set_xlabel(r"$n_0 (cm^{-3})$", fontsize = 12)
            xl = r"$n_0 = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}} cm^{{-3}}$".format(x_pers[1], 
                                                                               x_pers[2] - x_pers[1], 
                                                                               x_pers[1] - x_pers[0])

        _ = g.ax_joint.set_ylabel(r"$H_r$ (kpc)", fontsize = 12)
        yl = r"$H_r = {0:.2f}^{{{1:.2f}}}_{{{2:.2f}}}$ kpc".format(y_pers[1], 
                                                                  y_pers[2] - y_pers[1], 
                                                                  y_pers[1] - y_pers[0])


        _ = g.ax_joint.text(xlim[1], ylim[1], 
                          "{}\n{}\n{}".format(self.ion,xl, yl), 
                          fontsize = 12, 
                          ha = "right", 
                          va = "top")

        _ = g.ax_joint.set_xlim(xlim)
        _ = g.ax_joint.set_ylim(ylim)

        # _ = g.ax_joint.set_title("{}\n{}".format(xl, yl), fontsize = 12)
        plt.tight_layout()

        return g









