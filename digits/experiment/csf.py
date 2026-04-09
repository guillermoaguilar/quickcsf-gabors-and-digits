#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSF  and psychometric function implementation following Lesmes et al. (2010).

@author: G. Aguilar, Feb 2025

"""
import numpy as np


def csf(f, gamma_max, f_max, beta, delta):
    """ CSF parametrized as truncated parabola.

    Implements Eq. 1 and 2 in Lesmes et al. (2010).

    Either f is a vector and other parameters are floats
    OR
    parameters are vectors and f is a float.
    
    f is given in linear units

    Retuns log sensitivity.
    """

    k = np.log10(2)
    beta_p = np.log10(2*beta)

    log_f = np.log10(f)
    log_f_max = np.log10(f_max)
    log_g_max = np.log10(gamma_max)

    s_prime = log_g_max - k*((log_f - log_f_max)/(beta_p/2))**2
    s = np.copy(s_prime)

    if isinstance(f, float): # f is a number, so parameters are vectors, we simply substract
        cutoff = np.array(gamma_max - delta)
    else: # now f is a vector
        cutoff = np.repeat(gamma_max - delta, len(f))

    cutoff[cutoff < 1e-8] = 1e-8 # to avoid numerical problems

    ####
    ind = np.logical_and(f < f_max,  s_prime < np.log10(cutoff))
    s[ind] = np.log10(cutoff[ind])

    return s


def aulcsf(f, gamma_max, f_max, beta, delta):
    """Returns the area under the curve of the log sensitivity for given parameters
    
    The argument f should be in linear units, the function transform it to log 
    units automatically
    """

    logS = csf(f, gamma_max, f_max, beta, delta)
    return np.trapz(y=logS, x=np.log10(f))



def psi(c, f, gamma_max, f_max, beta, delta, slope=2, guessrate=0.5, lapserate=0.03):
    """ Probability correct for a given stimulus pair c, f
        and some CSF parameters.
        Implements Weibull function, as definied in Schutt et al. (2016) (psignifit)
    """

    logS = csf(f, gamma_max, f_max, beta, delta)

    s = np.power(10, logS)
    s[s < 1e-8] = 1e-8  # to avoid numerical issues, we truncate sensitivity at 10^-8
    t = 1/s # threshold, in linear units

    # Weibull sigmoid
    C = np.log(-np.log(0.05)) - np.log(-np.log(.95))
    e = C * (np.log(c) - np.log(t))/(slope)
    sigmoid = 1 - np.exp(np.log(.5)*np.exp(e))

    return guessrate + (1 - guessrate - lapserate)*sigmoid


if __name__=="__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk')

    # %%
    parameters={'gamma_max': 140,
                'f_max': 3.82,
                'beta': 0.24,
                'delta': 125}

    # %% plots CSF
    f = np.logspace(np.log10(0.01), np.log10(36), 1000)
    log_sensitivity = csf(f, **parameters)
    area = aulcsf(f, **parameters)

    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    axes.fill_between(f, y1=10**log_sensitivity, color=(0, 0, 0.8, .5))

    axes.set_xlabel('Spatial frequency (CPD)')
    axes.set_xscale('log')
    axes.set_xlim((.3, 30))
    axes.set_xticks([0.3, 1, 3, 10, 30])
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    axes.set_yscale('log')
    axes.set_ylim((.5, 300))
    axes.set_yticks([1, 10, 100])
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_ylabel('Sensitivity (1/contrast)')

    p = parameters['gamma_max'], parameters['f_max'], parameters['beta'], parameters['delta']
    axes.set_title("$\\gamma_{max} = %d, f_{max} = %.1f, \\beta = %.1f, \\delta = %d$" % p)
    axes.annotate(f"AULCSF = {area.round(2)}", xy=(0.5, 0.1), xycoords='axes fraction', color='w')
    sns.despine()

    plt.show()

    # %% tests psychometric function for a vector of frequencies and  contrasts,
    # covering all combinations
    frequencies = np.logspace(np.log10(.2), np.log10(36), 25, endpoint=True)
    contrasts = np.logspace(np.log10(0.001), np.log10(1), 25, endpoint=True)

    stimulus_space = np.array([[c, f] for f in frequencies for c in contrasts])

    c = stimulus_space[:, 0]
    f = stimulus_space[:, 1]

    prob = psi(c, f, **parameters)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.plot(contrasts, prob.reshape((25, 25)).T)
    axes.set_xscale('log')
    axes.set_ylim(0.5, 1)
    plt.show()

    # %% we also need psi to work for a fixed value of contrast and frequency,
    # but for a vector of the other parameters
    c = 0.1
    f = 0.2

    gamma_max_vector = np.logspace(np.log10(2),np.log10(2000), 25, endpoint=True)

    f_max_vector = np.logspace(np.log10(.2),
                               np.log10(20),
                               25, endpoint=True)

    beta_vector = np.logspace(np.log10(0.2),
                              np.log10(10),
                              25, endpoint=True)

    delta_vector = np.logspace(np.log10(0.02), 2, 25, endpoint=True)

    param_space = np.array([[gamma_max, fmax, beta, delta] for gamma_max in gamma_max_vector for fmax in f_max_vector for beta in beta_vector for delta in delta_vector])

    prob = psi(c,
               f,
               gamma_max=param_space[:,0],
               f_max=param_space[:,1],
               beta=param_space[:,2],
               delta=param_space[:,3])

    print(prob)
    plt.plot(prob, '.')
    plt.show()


    
