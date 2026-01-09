#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the quick CSF algorithm.
Lesmes et al. (2010). Bayesian adaptive estimation of the contrast sensitivity
function: The quick CSF method. Journal of vision, 10(3), 17-17.

Assumes a 4-parameter model of CSF (truncated parabola):
    Peak sensitivity (gamma_max): highest contrast sensitivity
    Peak frequency (f_max): spatial frequency at which peak sensitivity occurs
    Bandwidth (beta): full-width at half-maximum of the parabola
    Delta (delta): difference between peak sensitivity and truncation at low-frequencies


Notes:
    This implementation uses the maximum a posteriori as estimator, not
    the mean, as in Lesmes et al.
    
    This implementation has two types of priors, that can be set with parameter
    `priors` when instantiating the object qCSF. The option `prios='flat'`
    defined completely flat priors. This is not what is implemented in the 
    original paper. Reason for this is that I dont get to reproduce the exact
    priors from the formulas and values in the paper. 
    The option `priors='sech' is the closest approximation to the priors in the 
    paper, but this was done adjusting the parameters 'by eye' so that the
    marginals and Fig. A1 would more or less coincide.


@author: G. Aguilar, Feb 2025
"""
import numpy as np
import csv

from csf import csf, psi, aulcsf


def entropy(p):
    """ Entropy definition following Eq. A2"""
    return -p*np.log(p) - (1-p)*np.log(1-p)


def sech(z):
    """ Hyperbolic secant function"""
    return 2.0/(np.exp(z) + np.exp(-z))


def prior(theta, theta_confidence, theta_guess):
    """ Returns prior accoriding to Eq. A1"""
    return normalize(sech(theta_confidence * (theta - theta_guess)))


def flat_prior(n):
    """ Returns a flat prior"""
    return normalize(np.ones(n))


def normalize(x):
    return x/np.sum(x)


def build_4dvolume(a, b, c, d):
    a = a[:, np.newaxis, np.newaxis, np.newaxis]
    b = b[np.newaxis, :, np.newaxis, np.newaxis]
    c = c[np.newaxis, np.newaxis, :, np.newaxis]
    d = d[np.newaxis, np.newaxis, np.newaxis, :]

    z = a*b*c*d
    return z


class qCSF():
    def __init__(self, frequency_vector=None, contrast_vector=None,
                 gamma_max_vector=None, f_max_vector=None,
                 beta_vector=None, delta_vector=None,
                 priors='flat'):
        """ Initializes the quickCSF object"""

        # set default parameter vectors
        if frequency_vector is None:
            frequency_vector = np.logspace(np.log10(.2),
                                           np.log10(36),
                                           25, endpoint=True)

        if contrast_vector is None:
            contrast_vector = np.logspace(np.log10(.001),
                                          np.log10(1),
                                          25, endpoint=True)

        self.frequency_vector = frequency_vector
        self.contrast_vector = contrast_vector

        # stimulus space
        self.stimulus_space = np.array([[c, f]
                                        for f in self.frequency_vector
                                        for c in self.contrast_vector])

        # initialize priors
        # x value vectors for priors.
        if gamma_max_vector is None:
            gamma_max_vector = np.logspace(np.log10(2),
                                           np.log10(2000),
                                           40, endpoint=True)
        if f_max_vector is None:
            f_max_vector = np.logspace(np.log10(.2),
                                       np.log10(20),
                                       40, endpoint=True)

        if beta_vector is None:
            beta_vector = np.logspace(np.log10(.2),
                                      np.log10(10),
                                      40, endpoint=True)

        if delta_vector is None:
            delta_vector = np.logspace(np.log10(0.02),
                                       np.log10(200),
                                       40, endpoint=True)

        self.gamma_max_vector = gamma_max_vector
        self.f_max_vector = f_max_vector
        self.beta_vector = beta_vector
        self.delta_vector = delta_vector


        self.param_space = np.array([[gamma_max, fmax, beta, delta]
                                     for gamma_max in self.gamma_max_vector
                                     for fmax in self.f_max_vector
                                     for beta in self.beta_vector
                                     for delta in self.delta_vector])

        # prior values, as flat pdfs
        if priors == 'flat':
            self.p_gamma_max = flat_prior(len(self.gamma_max_vector))
            self.p_f_max = flat_prior(len(self.f_max_vector))
            self.p_beta = flat_prior(len(self.beta_vector))
            self.p_delta = flat_prior(len(self.delta_vector))

        elif priors == 'sech':
            # ATTENTION
            # priors here are not exactly the same as in the paper.
            # using the same ones give very peaked and not flat marginals.
            # either there is missing information in the paper, or
            # the implementation here of function prior() is wrong.
            # I adjusted the values to look more or less similar to figure A1
            self.p_gamma_max = prior(self.gamma_max_vector,
                                     theta_confidence=0.001, theta_guess=100)
            self.p_f_max = prior(self.f_max_vector,
                                 theta_confidence=0.1, theta_guess=2.5)
            self.p_beta = prior(self.beta_vector,
                                theta_confidence=0.25, theta_guess=3)
            self.p_delta = prior(self.delta_vector,
                                theta_confidence=0.005, theta_guess=np.log10(0.5))

        # joint prior, 4d volume
        self.p = build_4dvolume(self.p_gamma_max, self.p_f_max,
                                self.p_beta, self.p_delta)
        self.p = normalize(self.p)

        # saving dimensionality
        self.shape = self.p.shape

        # flattening
        self.p = self.p.flatten()

        # fixed parameters of psychomeric function
        self.lapserate = 0.04  # lapse rate
        self.guessrate = 0.04
        self.slope = 2  # slope parameter, renamed from beta to slope, to avoid confusion

        self.current_frequency = None
        self.current_contrast = None

        # log
        self.history = [] # contrast, frequency, response
        
        # frequency vector to evaluate CSF and save results
        self.f = np.logspace(np.log10(0.3), np.log10(30), 100)
        

    def next_stimulus(self):
        """Determine the next stimulus to be presented"""
        nsamples = 100
        nfraction = 0.1  # the top fraction from where to randomly take the next stimulus

        # randomly sample from parameter space
        indices = np.random.choice(np.arange(len(self.p)),
                                   size=nsamples,
                                   p=self.p)

        # iterate through sample, computing information gain
        prob = np.zeros((nsamples, self.stimulus_space.shape[0]))
        for j, i in enumerate(indices):

            gamma_max, f_max, beta, delta = self.param_space[i]

            # all stimuli, defined in contrast and frequency
            c = self.stimulus_space[:, 0]
            f = self.stimulus_space[:, 1]

            # get prob. correct for each
            prob[j, :] = psi(c, f, gamma_max, f_max, beta, delta,
                             slope=self.slope, guessrate=self.guessrate,
                             lapserate=self.lapserate)

        # we now calculate the entropy and information gain for each sampled stimulus
        pbar = np.sum(prob, axis=0)/nsamples
        hbar = np.sum(entropy(prob), axis=0)/nsamples
        gain = entropy(pbar)-hbar

        # sort by gain
        i_sorted_gain = np.argsort(-gain)

        # choose randomly one of the first 10%)
        i_r = np.random.choice(range(int(nfraction*nsamples)))
        index_stim = i_sorted_gain[i_r]

        # recover the values for the selected stimulus
        self.current_contrast  = self.stimulus_space[index_stim, 0]
        self.current_frequency = self.stimulus_space[index_stim, 1]

        return self.current_contrast, self.current_frequency


    def add_response(self, response):
        """Record the observer's response to the last returned stimulus by
        next() and updates the parameter probabilities"""

        # we first get psi(theta) for all possible parameters,
        # but for the given contrast and frequency presented

        f = self.current_frequency
        c = self.current_contrast

        prob = psi(c,
                   f,
                   gamma_max=self.param_space[:, 0],
                   f_max=self.param_space[:, 1],
                   beta=self.param_space[:, 2],
                   delta=self.param_space[:, 3])

        # we update probabilities
        if response: # is 1
            self.p = self.p * prob
        else:
            self.p = self.p * (1 - prob)

        # Normalize probabilities
        self.p = normalize(self.p)

        # Save history
        self.history.append([c, f, response])

    def get_p_volume(self):
        """ Returns the volume containing the probability over the 4 parameters"""
        return self.p.reshape(self.shape)

    def get_estimates(self):
        """ Returns the estimated parameters, calculated as the maximum of the posterior"""
        volume = self.get_p_volume()
        
        # TODO: add option for mean a posteriori, not max a posteriori
        global_max = np.unravel_index(np.argmax(volume), volume.shape)

        return  {'gamma_max': self.gamma_max_vector[global_max[0]],
                 'f_max': self.f_max_vector[global_max[1]],
                 'beta' : self.beta_vector[global_max[2]],
                 'delta': self.delta_vector[global_max[3]]
                }
    
    def sample_posterior(self, nsamples):
        """
        Samples the posterior according to its probability density. 
        
        This is required to calculate confidence intervals.

        Parameters
        ----------
        nsamples : int,
            Number of samples.

        Returns
        -------
        samples : np.array
            Array of size (nsamples x 4) containing the sampled parameters.

        """
        
        # getting samples from the posterior 
        samples_idx = np.random.choice(np.arange(len(self.p)),
                                       size=(nsamples),
                                       replace=True,
                                       p = self.p)
        
        samples = self.param_space[samples_idx]
        
        return samples
    
    
    def get_confidence_intervals(self, alpha=0.05, nsamples=1000):
        """
        Returns confidence intervals calculated from samples from the posterior

        Parameters
        ----------
        alpha : float, optional
            Confidence leval. The default is 0.05, for 95% confidence.
        nsamples : int, optional
                Number of samples. The default is 1000.

        Returns
        -------
        S_low : np.array
            Lower bound of confidence for the log sensitivity
        S_high : np.array
            Higher bound of confidence for the log sensitivity

        """
        samples = self.sample_posterior(nsamples)
        
        # evaluating each sample
        S_samples = np.zeros((nsamples, len(self.f)))
        
        for i in range(nsamples):
            S_samples[i, :] = csf(self.f,
                                  gamma_max=samples[i, 0], 
                                  f_max=samples[i, 1],
                                  beta=samples[i, 2],
                                  delta=samples[i, 3])
        
            
        S_samples = np.sort(S_samples, axis=0) # sort each column independently        
        
        low_cut = int(nsamples*(alpha/2))
        high_cut = int(nsamples*(1 - alpha/2))
        
        S_low = S_samples[low_cut, :]
        S_high = S_samples[high_cut, :]
        
        
        return S_low, S_high


    def get_estimate_csf(self):
        """
        Returns the estimated CSF with its confidence interval.

        Returns
        -------
        S_estimate : np.array
            CSF evaluated on the point estimate.
        S_low : np.array
            Lower confidence bound for the CSF
        S_high : np.array
            Higher confidence bound for the CSF

        """
        estimate = self.get_estimates()
        
        # evaluates point estimate
        logS_estimate = csf(self.f, **estimate)
              
        # get CIs, already evaluated 
        logS_low, logS_high = self.get_confidence_intervals()
        
        
        S_estimate = np.power(10, logS_estimate)
        S_low = np.power(10, logS_low)
        S_high = np.power(10, logS_high)
        
        
        return S_estimate, S_low, S_high
    
        
    def save_results(self, filename):
        """
        Save CSF's estimate and its confidence interval in a CSV file.

        Parameters
        ----------
        filename : str
            Filename to be used for saving. File extensions will be appended.

        Returns
        -------
        None.

        """
        
        ### save history
        #with open(filename + '_history.csv', 'w') as csv_file:  
        #    writer = csv.writer(csv_file)
        #    writer.writerow(['contrast', 'sf', 'response'])
        #    for c, sf, r in self.history:
        #       writer.writerow([c, sf, r])
        
        ### gets parameter estimates
        estimate = self.get_estimates()

        # get evaluated CSF and its CI        
        S_estimate, S_low, S_high = self.get_estimate_csf()
        
        # stack for easy parsing to CSV
        data = np.vstack((self.f, S_estimate, S_low, S_high)).T
                
        # saves the parameter estimates in a CSV file
        with open(filename + '_estimates.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in estimate.items():
               writer.writerow([key, value])
               
        # saves the evaluated CSF and CIs
        np.savetxt(filename + '_csf.csv', 
                   data,
                   delimiter=',',
                   header='frequency,sensitivity,low_bound_sensitivity,high_bound_sensitivity',
                   )
    
        # dumps also the posterior volume
        np.savez(filename + '_pvolume.npz', 
                 p=self.get_p_volume(), 
                 allow_pickle=False)
        
              
        
        
    

if __name__=="__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import  seaborn as sns
    sns.set_context('notebook')


    qcsf = qCSF()
    
    
    

    # ## plotting probabilities
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
    axes[0][0].plot(qcsf.gamma_max_vector, qcsf.p_gamma_max)
    #axes[0][0].set_xlabel('Octaves')
    axes[0][0].set_xscale('log')
    #axes[0][0].set_xlim((10, 1000))
    axes[0][0].set_xticks([10, 30, 100, 300, 1000])
    axes[0][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0][0].set_ylim((0, 1))
    axes[0][0].set_title(r'$\gamma_{max}$')
    axes[0][0].set_ylabel('Probability')

    axes[0][1].plot(qcsf.f_max_vector, qcsf.p_f_max)
    axes[0][1].set_xlabel('Spatial frequency (CPD)')
    axes[0][1].set_xscale('log')
    #axes[0][1].set_xlim((.3, 30))
    axes[0][1].set_xticks([0.3, 1, 3, 10, 30])
    axes[0][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0][1].set_ylim((0, 1))
    axes[0][1].set_title(r'$f_{max}$')


    axes[1][0].plot(qcsf.beta_vector, qcsf.p_beta)
    axes[1][0].set_xlabel('Octaves')
    axes[1][0].set_xscale('log')
    #axes[1][0].set_xlim((.3, 20))
    axes[1][0].set_xticks([0.3, 1, 3, 10])
    axes[1][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1][0].set_ylim((0, 1))
    axes[1][0].set_title(r'$\beta$')
    axes[1][0].set_ylabel('Probability')


    axes[1][1].plot(qcsf.delta_vector, qcsf.p_delta)
    axes[1][1].set_xlabel('Log units')
    axes[1][1].set_xscale('log')
    #axes[1][1].set_xlim((.05, 2))
    axes[1][1].set_xticks([0.05, .2, .6, 2])
    axes[1][1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1][1].set_ylim((0, 1))
    axes[1][1].set_title(r'$\delta$')

    sns.despine()
    plt.show()


    c,f = qcsf.next_stimulus()
    
    qcsf.add_response(1)
    
    
