# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    pass
    #data_path = "../data/"
# ------------------------------


# Our main plotting package (must have explicit import of submodules)
import bokeh.io
import bokeh.plotting

# Enable viewing Bokeh plots in the notebook
bokeh.io.output_notebook()


import numpy as np
import pandas as pd
import pandas
import math

import bebi103
import iqplot

from matplotlib import pyplot as plt
import tqdm

import scipy.optimize
import scipy.stats as st
import scipy.special
import scipy
# -

data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"


def loglik_gamma(params, t):
    '''Log likelihood function for the gamma distribution. 
            Takes alpha and beta and returns the negative sum of the logpdf from scipy.stats.gamma
    Input
    --------
    Params: list
        [alpha, beta]
    t: 1D list
        contains times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    Returns
    ---------
    result: float
        negative sum of log likelihood, summed over each timepoint in t
    
    '''
    
    #extract alpha and beta
    alpha, beta = params
    
    #convert to floats, important because dataframes hold as strings
    alpha = float(alpha)
    beta = float(beta)
    
    
    #nonlogical to have negative alpha
    if alpha < 0:
        return np.inf
    
    #nonlogical to have negative beta
    if beta <0:
        return np.inf
    
    #just use scipy.stats.gamma function
    result = -np.sum(st.gamma.logpdf(t, alpha, loc=0, scale=1/beta))
    
    return result


def powell_res_gamma(t):
    '''Runs optimizer using Powell's method and loglik_gamma to find MLEs for alpha and beta for a given
            microtubule catastrophe set
    
    Input
    --------
    t: 1D list
        contains times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    Returns
    ---------
    res.x: numpy array
        [MLE_for_alpha, MLE_for_beta]
    '''

    res = scipy.optimize.minimize(
        fun=loglik_gamma,
        x0=[1,1],
        args=(np.transpose(t),),
        method='Powell',
        tol = 1e-10
    )

    return res.x


def gen_gamma(params, size, rg = np.random.default_rng()):
    '''Draw a bootstrap samples from Gamma distribution parameterized by params
        
        Input:
        -----------
        params: list
            [alpha_mle, beta_mle]
        size: int
            specifies length of the generated data
        rg: numpy random number generator object
            Feels silly to have it here, but a bebi103 function requires it this way
        
        Returns: 
        ---------
            samples: numpy array 
                contains "size" # of samples randomly drawn from the gamma distribution parameterized 
                    by params  
    '''
    
    alpha, beta = params
    samples = rg.gamma(alpha, 1/beta, size)
    return samples


def draw_bs_reps_mle_gamma(mle_fun, data, params, size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Input
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.notebook.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), params) for _ in iterator])


def powell_reps_gamma(t):
    '''Computes bootstrapped replicates (MLEs) for gamma distribution
    
    Input
    -----------
    t: 1D list
        contains times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
        
    Returns
    -----------
    ble_reps: list of arrays
        [array([alpha_mle_1, beta_mle_1]),
         array([alpha_mle_2, beta_mle_2]),
         array([alpha_mle_3, beta_mle_3]),
                    ...              
         array([alpha_mle_n, beta_mle_n])]
    '''
    
    #get MLEs for alpha and beta
    alpha, beta = powell_res_gamma(np.transpose(t))

    #number of replicates
    size = 100
    
    ble_reps = []
    for i in tqdm.tqdm(range(int(size))):
        
        #generate sample dataset from gamma distribution 
        g = gen_gamma([alpha, beta], size=len(t))
        
        #calculate MLE for sample dataset
        m = powell_res_gamma(g)

        #add MLEs to big list
        ble_reps.append(m)

    return ble_reps


def loglik_2beta(beta1, t):
    """Compute the log likelihood for a given value of β1,
    assuming Δβ is set so that the dervitive of the log
    likelihood with respect to β1 vanishes."""
    beta1 = float(beta1)
    n = len(t)
    tbar = np.mean(t)
    beta1_tbar = beta1 * tbar
    
    if beta1_tbar > 2 or beta1_tbar < 1:
        return np.nan

    if np.isclose(beta1_tbar, 2):
        return -2 * n * (1 + np.log(tbar) - np.log(2)) + np.sum(np.log(t))
        
    if np.isclose(beta1_tbar, 1):
        return -n * (1 + np.log(tbar))
            
    delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)

    ell = n * (np.log(beta1) + np.log(beta1 + delta_beta) - np.log(delta_beta))
    ell -= n * beta1_tbar
    ell += np.sum(np.log(1 - np.exp(-delta_beta * t)))
    
    return ell


def dlog_like_dbeta1(beta1, t):
    """Returns the derivative of the log likelihood w.r.t. Δβ
    as a function of β1, assuming Δβ is set so that the dervitive 
    of the log likelihood with respect to β1 vanishes."""
    n = len(t)
    tbar = np.mean(t)
    beta1_tbar = beta1 * tbar
    
    if beta1_tbar > 2 or beta1_tbar < 1:
        return np.nan

    if np.isclose(beta1_tbar, 2) or np.isclose(beta1_tbar, 1):
        return 0.0
            
    delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)
    
    exp_val = np.exp(-delta_beta * t)
    sum_term = np.sum(t * exp_val / (1 - exp_val))
    
    return -n / delta_beta + n / (beta1 + delta_beta) + sum_term


def draw_beta_model(betas, size, rg):
    beta1, beta2 = betas
    return rg.exponential(1 / beta1, size=size) + rg.exponential(1 / beta2, size=size)


def mle_two_step(t, nbeta1=500):
    """Compute the MLE for the two-step model."""
    # Compute ∂ℓ/∂Δβ for values of beta_1
    tbar = np.mean(t)
    beta1 = np.linspace(1 / tbar, 2 / tbar, nbeta1)
    deriv = np.array([dlog_like_dbeta1(b1, t) for b1 in beta1])
    
    # Add the roots at the edges of the domain
    beta1_vals = [1 / tbar, 2 / tbar]
    ell_vals = [loglik_2beta(beta1_vals[0], t), loglik_2beta(beta1_vals[1], t)]
    
    # Find all sign flips between the edges of the domain
    sign = np.sign(deriv[1:-1])
    inds = np.where(np.diff(sign))[0]
    
    # Perform root finding at the sign flips
    for i in inds:
        b1 = scipy.optimize.brentq(dlog_like_dbeta1, beta1[i+1], beta1[i+2], args=(t,))
        beta1_vals.append(b1)
        ell_vals.append(loglik_2beta(b1, t))
        
    # Find the value of beta1 that gives the maximal log likelihood
    i = np.argmax(ell_vals)
    beta1 = beta1_vals[i]

    # Compute beta 2
    if np.isclose(beta1, 1 / tbar):
        delta_beta = np.inf
    else:
        delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)

    beta2 = beta1 + delta_beta
    
    return np.array([beta1, beta2])


def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


def gamma_qq(alpha_mle, beta_mle, time, var_str, choice):
    '''Generates plot for graphical model comparison for gamma distribution
    
    Input
    ----------
    alpha_mle: float
        MLE for alpha
    
    beta_mle: float
        MLE for beta
    
    time: list
        contains list of times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    var_str: string
        name for graph, in particular should be a number, like '12' to represent tubulin concentration
        
    choice: string
        'diff' --> shows difference between original and theoretical 
        'fill'  --> predictive ECDF with opaque fill between 97.5th and 2.5th percentiles
        'contour' -->  similar to 'fill' but wit levels to show confidence and a bit prettier
        'qq' --> a QQ plot 
        

    Return
    ----------
    s: bokeh figure
        returning figure object for potential dashboarding applications
    '''
    
    #generate sample data from the MLEs
    gamma_samples = np.array(
        [gen_gamma([alpha_mle, beta_mle], size=len(time)) for _ in range(1000)]
        )
   
    #plot predictive ECDF difference from original data
    if choice == 'diff':
        s = bebi103.viz.predictive_ecdf(
            samples=gamma_samples, 
            data=time, 
            diff='ecdf', 
            discrete=True, 
            x_axis_label="time"
            )
        s.title= 'Gamma ECDF difference ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')

    #plot predictive ECDF fill
    if choice == 'fill':
        n_theor = np.arange(0, gamma_samples.max() + 1)
        ecdfs = np.array([ecdf(n_theor, sample) for sample in gamma_samples])
        ecdf_low, ecdf_high = np.percentile(ecdfs, [2.5, 97.5], axis=0)
        s = bebi103.viz.fill_between(
            x1=n_theor,
            y1=ecdf_high,
            x2=n_theor,
            y2=ecdf_low,
            patch_kwargs={"fill_alpha": 0.5},
            x_axis_label="n",
            y_axis_label="ECDF",
            )
        s = iqplot.ecdf(data=time, palette=["orange"], p=s)
        s.title = 'Gamma predictive ECDF fill ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')
     
    #plot predictive ECDF with contour colors for confidence level
    if choice == 'contour':
        s = bebi103.viz.predictive_ecdf(
                samples=gamma_samples, 
                data=time, 
                discrete=True, 
                x_axis_label="Time to catastrophe (s)"
                )
        s.title= 'Gamma predictive ECDF ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')
    
    #plot QQ plot
    if choice == 'qq':
        s = bebi103.viz.qqplot(
            data=time,
            samples=gamma_samples,
            x_axis_label="??",
            y_axis_label="??",
            )
    
        s.title= 'Gamma QQ ' + var_str
        s.yaxis.axis_label = 'Time to catastrophe (s)'
        s.line([0], [0], color = 'grey', legend_label = 'observed')
    
    s.line([0], [0], color = '#3d85c6', legend_label = 'predicted')
    s.legend.location = "bottom_right"    
    s.xaxis.axis_label = 'Time to catastrophe (s)'
    
    return s
    


def two_beta_qq(beta1_mle, beta2_mle, time, var_str, choice):
    '''Generates plot for graphical model comparison for 2-beta distribution
    
    Input
    ----------
    beta1_mle: float
        MLE for beta1
    
    beta2_mle: float
        MLE for beta2
    
    time: list
        contains list of times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    var_str: string
        name for graph, in particular should be a number, like '12' to represent tubulin concentration
        
    choice: string
        'diff' --> shows difference between original and theoretical 
        'fill'  --> predictive ECDF with opaque fill between 97.5th and 2.5th percentiles
        'contour' -->  similar to 'fill' but wit levels to show confidence and a bit prettier
        'qq' --> a QQ plot 
        
    
    Return
    ----------
    s: bokeh figure
        returning figure object for potential dashboarding applications
    '''
    
    #generate sample data from the MLE
    two_beta_samples = np.array(
        [draw_beta_model([beta1_mle, beta2_mle],len(time), rg = np.random.default_rng()) for _ in range(1000)]
        )
   
    #plot predictive ECDF difference from original data
    if choice == 'diff':
        s = bebi103.viz.predictive_ecdf(
            samples=two_beta_samples, data=time, diff='ecdf', discrete=True, x_axis_label="time"
            )
        s.title= '2-beta ECDF difference ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')
    
    #plot predictive ECDF fill
    if choice == 'fill':
        n_theor = np.arange(0, two_beta_samples.max() + 1)
        ecdfs = np.array([ecdf(n_theor, sample) for sample in two_beta_samples])
        ecdf_low, ecdf_high = np.percentile(ecdfs, [2.5, 97.5], axis=0)
        s = bebi103.viz.fill_between(
            x1=n_theor,
            y1=ecdf_high,
            x2=n_theor,
            y2=ecdf_low,
            patch_kwargs={"fill_alpha": 0.5},
            x_axis_label="n",
            y_axis_label="ECDF",
            )
        s = iqplot.ecdf(data=time, palette=["orange"], p=s)
        s.title = '2-beta predictive ECDF fill ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')
    
    #plot predictive ECDF with contour colors for confidence level
    if choice == 'contour':
        s = bebi103.viz.predictive_ecdf(
                samples=two_beta_samples, 
                data=time, 
                discrete=True, 
                x_axis_label="Time to catastrophe (s)"
                )
        s.title= '2-beta predictive ECDF ' + var_str
        s.line([0], [0], color = 'orange', legend_label = 'observed')
    
    #plot QQ plot
    if choice == 'qq':
        s = bebi103.viz.qqplot(
            data=time,
            samples=two_beta_samples,
            x_axis_label="??",
            y_axis_label="??",
            )
        s.line([0], [0], color = 'grey', legend_label = 'observed')
        s.title= '2-beta QQ ' + var_str
        s.yaxis.axis_label = 'Time to catastrophe (s)'
    
    s.line([0], [0], color = '#3d85c6', legend_label = 'predicted')
    s.legend.location = "bottom_right"    
    s.xaxis.axis_label = 'Time to catastrophe (s)'
    
    return s
    


def dict_maker(var_str, var_cf, var_mean):
    '''makes a dictionary in the correct format for bebi103.viz.confint
       but not using for bebi103.viz.confint
       
       Input
       ---------
       var_str: string
           name of variable
           will appear as plot label for variable in bebi103.viz.confint
        
        var_cf: list
            should be [2.5%, 97.5%] of bs_reps 
        
        var_mean: float
            either mean of bs_reps or the MLE
            
        Return
        -----------
        my_dict: dictionary
            conf_int: var_cf
            estimate: var_mean
            label: var_str
            
        output is formatted for use with bebi103.viz.confint     
       '''
    
    dict_string = '{}'.format(var_str)
    my_dict = {'conf_int': (var_cf)}
    my_dict['estimate'] = var_mean
    my_dict['label'] = dict_string
    
    return my_dict


def plot_bs_reps(bs_reps, x_str, y_str, title_str):
    
    '''Generates scatterplot of bs_reps parameters
    
    Input
    ----------
    bs_reps: *np array* of lists in format:
            np.array([[parameter1_1, parameter2_1],
                      [parameter1_2, parameter2_2],
                      [parameter1_3, parameter2_3],
                                    .....
                      [parameter1_n, parameter2_n]])
    
    x_str: string
        label for x-axis
    
    y_str: string
        label for y-axis
    
    title_str: string
        title for plot
        
    Return
    ---------------
    p: bokeh.Figure object
    
    also shows p
    '''
    
    
    
    
    # Make figure
    p = bokeh.plotting.figure(
        width=400,
        height=300,
        x_axis_label=x_str,
        y_axis_label=y_str,
        title = title_str
    )
    bs_reps_noinf = []
    for i in bs_reps:
        if i[1] == np.inf:
            bs_reps_noinf.append([i[0], 10000])
        else:
            bs_reps_noinf.append([i[0], i[1]])
    
    bs_reps_noinf = np.array(bs_reps_noinf)
            
    # Add glyphs
    p.circle(
        x=bs_reps_noinf[:,0],
        y=bs_reps_noinf[:,1],
        #legend_label="normal sleepers",
    )
    
    #p.add_layout(p.legend[0], 'right')
    #bokeh.io.show(p)
    
    return p


def inf_counter(bs_reps):
    '''
    Counts infinites in 2nd parameter of bs_reps
    
    Input
    ----------
    bs_reps: *np array* of lists in format:
            np.array([[parameter1_1, parameter2_1],
                      [parameter1_2, parameter2_2],
                      [parameter1_3, parameter2_3],
                                    .....
                      [parameter1_n, parameter2_n]])
    
    Returns
    ----------
    count: int
        number of times the second parameter is infinity            
    '''
    
    count = 0
    for i in bs_reps:
        if i[1] == np.inf:
            count +=1
    return count


def two_beta_mle_computer(time, name_str):
    
    '''Computes MLEs, bootstrap replicates of MLEs, and confidence intervals for the 2-beta model
    
    Input
    ----------
    time: list
        contains list of times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    name_str: string
        label/condition of data (ie labeled vs unlabeled tubulin, tubulin concentration)
    
    Return
    -----------
    name_str: string
        exactly as input
    
    beta1_mle: float
        MLE for beta1
    
    beta2_mle: float
        MLE for beta2
    
    inf_percent: float
        % of beta2 that are infinite
    
    beta1_dict: dictionary
        conf_int: beta1 confidence interval (2.5%, 97.5%
        estimate: beta1 MLE
        label: 'beta1'
    
    beta2_dict: dictionary
        conf_int: beta2 confidence interval (2.5%, 97.5%
        estimate: beta2 MLE
        label: name_str
        
    bs_reps_two_step: list of lists
        [[beta1_1, beta2_1], 
         [beta1_2, beta2_2]
                      ....]
                      
    p: bokeh.Figure
        plot of bs_reps, x=beta1, y=beta2
    '''
    
    #get MLEs
    beta1_mle, beta2_mle = mle_two_step(time)
    
    #draw bootstrap replicate MLEs
    bs_reps_two_step = bebi103.bootstrap.draw_bs_reps_mle(
        mle_two_step, 
        draw_beta_model, 
        time, 
        size=100, 
        n_jobs=1
        )
    
    #count infinities
    inf_count = inf_counter(bs_reps_two_step)
    inf_percent = round(((inf_count/len(bs_reps_two_step))*100),3)
    #print('{}% of {} beta2 MLEs are infinity'.format(inf_percent, name_str))
    
    #plot the bootstrap replicates
    p = plot_bs_reps(np.array(bs_reps_two_step), 'beta1', 'beta2', '2-beta ' + name_str)
    
    #get upper and lower bounds of confidence intervals
    percs = np.percentile(bs_reps_two_step, [2.5, 97.5], axis=0)
    
    beta1_1, beta2_1 = percs[0]
    beta1_2, beta2_2 = percs[1]
    
    #put confidence intervals in a little list together
    beta1_conf = [beta1_1, beta1_2]
    beta2_conf = [beta2_1, beta2_2]
    
    #package the MLE results into a little dictionary
    beta1_dict = dict_maker(name_str, beta1_conf, beta1_mle)
    beta2_dict = dict_maker(name_str, beta2_conf, beta2_mle)
    
    return name_str, beta1_mle, beta2_mle, inf_percent, beta1_dict, beta2_dict, bs_reps_two_step, p


def gamma_mle_computer(time, name_str):

    '''Computes MLEs and bootstrap replicates of MLEs and confidence intervals for the gamma model
    
    
    Input
    ----------
    time: list
        contains list of times to microtubule catastrophe for one trial (ie 10 uM or 7 uM or labeled or unlabeled)
    
    name_str: string
        label/condition of data (ie labeled vs unlabeled tubulin, tubulin concentration)
    
    
    Return
    -----------
    name_str: string
        exactly as input
    
    alpha_mle: float
        MLE for alpha
    
    beta_mle: float
        MLE for beta
    
    alpha_dict: dictionary
        conf_int: alpha confidence interval (2.5%, 97.5%
        estimate: alpha MLE
        label: name_str
    
    beta_dict: dictionary
        conf_int: beta confidence interval (2.5%, 97.5%
        estimate: beta MLE
        label: name_str
        
    bs_reps_gamma: list of lists
        [[alpha_1, beta_1], 
         [alpha_2, beta_2]
                      ....]
                      
    p: bokeh.Figure
        plot of bs_reps, x=beta1, y=beta2
    
    
    '''
    
    #get MLEs
    alpha_mle, beta_mle = powell_res_gamma(time)
    
    #draw bootstrap replicate MLEs
    bs_reps_gamma = bebi103.bootstrap.draw_bs_reps_mle(
        powell_res_gamma, 
        gen_gamma, 
        time, 
        size=100, 
        n_jobs=1
        )
    
    #plot the bootstrap replicates
    p = plot_bs_reps(np.array(bs_reps_gamma), 'alpha', 'beta', 'Gamma ' + name_str)
    
    #get upper and lower bounds of confidence intervals
    percs = np.percentile(bs_reps_gamma, [2.5, 97.5], axis=0)
    alpha_1, beta_1 = percs[0]
    alpha_2, beta_2 = percs[1]
    
    #put confidence intervals in a little list together
    alpha_conf = [alpha_1, alpha_2]
    beta_conf = [beta_1, beta_2]
    
    #package the MLE results into a little dictionary
    alpha_dict = dict_maker(name_str, alpha_conf, alpha_mle)
    beta_dict = dict_maker(name_str, beta_conf, beta_mle)
    
    return name_str, alpha_mle, beta_mle, alpha_dict, beta_dict, bs_reps_gamma, p


