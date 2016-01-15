import numpy as np
import matplotlib.pyplot as plt
from emcee import PTSampler
import seaborn as sns
import itertools


class BayesPolynomialTest():
    def __init__(self, x, y, degrees=[1], ntemps=100, betamin=-6, nburn0=100,
                 nburn=100, nprod=100, nwalkers=100, unif_lim=100):
        self.x = x
        self.y = y
        self.y_std = np.std(y)
        self.ntemps = ntemps
        self.betas = np.logspace(0, betamin, ntemps)
        self.nburn0 = nburn0
        self.nburn = nburn
        self.nprod = nprod
        self.nwalkers = nwalkers
        self.stored_data = {}
        self.degrees = degrees
        self.max_degree = max(degrees)
        self.unif_lim = unif_lim

        for degree in degrees:
            self.fit_polynomial_p(degree)
        self.summarise_posteriors()

    def get_new_p0(self, sampler, ndim, scatter_val=1e-3):
        pF = sampler.chain[:, :, -1, :].reshape(
            self.ntemps, self.nwalkers, ndim)[0, :, :]
        lnp = sampler.lnprobability[:, :, -1].reshape(
            self.ntemps, self.nwalkers)[0, :]
        p = pF[np.argmax(lnp)]
        p0 = [[p + scatter_val * p * np.random.randn(ndim)
              for i in xrange(self.nwalkers)] for j in xrange(self.ntemps)]
        return p0

    def log_unif(self, x, a, b):
        if (x < a) or (x > b):
            return -np.inf
        else:
            return np.log(1./(b-a))

    def get_unif_prior_lims(self, key):
        if key == "sigma":
            return [1e-20*self.y_std, 2*self.y_std]
        if "b" in key:
            return [-self.unif_lim, self.unif_lim]

    def logp_polynomial_p(self, params):
        logps = [self.log_unif(param, *self.get_unif_prior_lims('b')) for
                 param in params[:-1]]
        logp = np.sum(logps)
        logp += self.log_unif(params[-1], *self.get_unif_prior_lims('sigma'))
        return logp

    def logl_polynomial_p(self, params, x, y):
        sigma = params[-1]
        resA = y - np.poly1d(params[:-1])(x)
        r = np.log(1/(sigma*np.sqrt(2*np.pi))*np.exp(-resA**2/(2*sigma**2)))
        return np.sum(r)

    def fit_polynomial_p(self, degree):
        name = 'p{}'.format(degree)
        self.stored_data[name] = {}
        ndim = degree + 2
        sampler = PTSampler(self.ntemps, self.nwalkers, ndim,
                            self.logl_polynomial_p, self.logp_polynomial_p,
                            loglargs=[self.x, self.y], betas=self.betas)
        param_keys = ["b{}".format(j) for j in range(0, degree+1)]
        param_keys.append('sigma')
        p0 = [[[np.random.uniform(*self.get_unif_prior_lims(key))
                for key in param_keys]
               for i in range(self.nwalkers)]
              for j in range(self.ntemps)]

        if self.nburn0 != 0:
            out = sampler.run_mcmc(p0, self.nburn0)
            self.stored_data[name]['chains0'] = sampler.chain[0, :, :, :]
            p0 = self.get_new_p0(sampler, ndim)
            sampler.reset()
        else:
            self.stored_data[name]['chains0'] = None

        out = sampler.run_mcmc(p0, self.nburn + self.nprod)
        self.stored_data[name]['chains'] = sampler.chain[0, :, :, :]

        self.stored_data[name]['sampler'] = sampler
        samples = sampler.chain[0, :, self.nburn:, :].reshape((-1, ndim))
        self.stored_data[name]['samples'] = samples

    def summarise_posteriors(self):
        for degree in self.degrees:
            name = "p{}".format(degree)
            b = [np.mean(self.stored_data[name]["samples"][:, i])
                 for i in range(degree+1)]
            self.stored_data[name]['b'.format(degree)] = b
            sigma = np.mean(self.stored_data[name]["samples"][:, -1])
            self.stored_data[name]['sigma'] = sigma

    def diagnostic_plot(self, fname="diagnostic.png", trace_line_width=0.05,
                        hist_line_width=1.5):

        fig = plt.figure(figsize=(8, 11))
        if self.ntemps > 1:
            nrows = self.max_degree + 4
        else:
            nrows = self.max_degree + 3

        colors = [sns.xkcd_rgb["pale red"],
                  sns.xkcd_rgb["medium green"],
                  sns.xkcd_rgb["denim blue"]]

        burn0s = np.arange(0, self.nburn0)
        prods = np.arange(self.nburn0, self.nburn0+self.nburn + self.nprod)

        ax00 = plt.subplot2grid((nrows, 2), (0, 0), colspan=2)
        ax00.plot(self.x, self.y, "o")
        x_plot = np.linspace(self.x.min(), self.x.max(), 100)
        ax00.set_xlabel("Data")

        Laxes = []
        Raxes = []
        for d in range(self.max_degree + 2):
            Lax = plt.subplot2grid((nrows, 2), (d+1, 0))
            Lax.set_xlabel(r"$\beta_{}$ posterior".format(d))
            Laxes.append(Lax)
            Rax = plt.subplot2grid((nrows, 2), (d+1, 1))
            Rax.set_xlabel(r"$\beta_{}$ trace".format(d))
            Raxes.append(Rax)
        Lax.set_xlabel(r"$\sigma$ posterior")
        Rax.set_xlabel(r"$\sigma$ trace")

        for i, degree in enumerate(self.degrees):
            data = self.stored_data["p{}".format(degree)]
            y = np.poly1d(data['b'])(x_plot)
            ax00.plot(x_plot, y, color=colors[i],
                      label="Poly of degree {}".format(degree))

            offset = self.max_degree - degree
            for j, d in enumerate(offset + np.arange(degree + 2)):
                ax = Laxes[d]
                ax.hist(data['samples'][:, j], bins=50,
                        linewidth=hist_line_width, histtype="step",
                        color=colors[i])

                ax = Raxes[d]
                if data['chains0'] is not None:
                    ax.plot(burn0s, data['chains0'][:, :, j].T,
                            lw=trace_line_width, color=colors[i])
                ax.plot(prods, data['chains'][:, :, j].T, lw=trace_line_width,
                        color=colors[i])
        ax00.legend(loc=2, frameon=False)

        for ax in Raxes:
            lw = 1.1
            ax.axvline(self.nburn0, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn, color="k", lw=lw, alpha=0.4)
            ax.axvline(self.nburn0+self.nburn+self.nprod, color="k",
                       lw=lw, alpha=0.4)

        if self.ntemps > 1:
            ax40 = plt.subplot2grid((nrows, 2), (nrows-1, 0), colspan=2)
            for i, d in enumerate(self.degrees):
                betas = self.betas
                alllnlikes = self.stored_data["p{}".format(d)][
                    'sampler'].lnlikelihood[:, :, self.nburn:]
                mean_lnlikes = np.mean(np.mean(alllnlikes, axis=1), axis=1)
                ax40.semilogx(betas, mean_lnlikes, "-o", color=colors[i])
                ax40.set_title("Linear thermodynamic integration")

        fig.tight_layout()
        fig.savefig(fname)

    def BayesFactor(self, print_result=True):
        evi_err = []
        for degree in self.degrees:
            name = "p{}".format(degree)
            sampler = self.stored_data[name]['sampler']
            lnevidence, lnevidence_err = sampler.thermodynamic_integration_log_evidence()
            log10evidence = lnevidence/np.log(10)
            log10evidence_err = lnevidence_err/np.log(10)
            evi_err.append((degree, log10evidence, log10evidence_err))

        if print_result:
            for mA, mB in itertools.combinations(evi_err, 2):
                mA_deg, mA_evi, mA_err = mA
                mA_name = "Poly of degree {}".format(mA_deg)
                mB_deg, mB_evi, mB_err = mB
                mB_name = "Poly of degree {}".format(mB_deg)
                bf = mB_evi - mA_evi
                bf_err = np.sqrt(mA_err**2 + mB_err**2)
                print "Bayes Factor ({}, {}) = {} +/- {}".format(
                    mB_name, mA_name, bf, bf_err)
