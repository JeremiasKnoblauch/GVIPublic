#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:06:46 2019

@author: jeremiasknoblauch

Description: DGP regression

"""


import sys
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(0)
import time

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.training import AdamOptimizer

#A new (robust) class of model-based likelihoods.
from robustified_likelihoods import betaDivGaussian, gammaDivGaussian

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from doubly_stochastic_dgp.dgp import DGP
from datasets import Datasets

#Path this file is in + /data
datasets = Datasets() 




"""Function serves three purposes: 
    (1) Reads relevant data/prepares the accurate split for index i
    (2) Calls 'DGP' for split i, which does the inference
    (3) Extracts and returns test performancee metrics to 'main'.
"""
def get_test_error(i, dataset, alpha, learning_rate=0.001, 
        iterations=20000, white = True, normalized = True, num_inducing = 100, 
        beta = None, gamma = None, div_weights = None):
    
    
    """STEP (1) Read in the data via the helpful 'Dataset' object"""
    data = datasets.all_datasets[dataset].get_data(seed=0, split=i, prop=0.9)
    X_train, Y_train, X_test, Y_test, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
    print('N: {}, D: {}, Ns: {}, Y_std: {}'.format(X_train.shape[0], X_train.shape[1], X_test.shape[0], Y_std))
    
    Z = kmeans2(X_train, num_inducing, minit='points')[0]

    #Dimensionality of X
    D = X_train.shape[1]

    # the layer shapes are defined by the kernel dims, so here all 
    # hidden layers are D dimensional 
    kernels = []
    for l in range(L):
        kernels.append(RBF(D))

    # between layer noise (doesn't actually make much difference but we include it anyway)
    for kernel in kernels[:-1]:
        kernel += White(D, variance=1e-5) 

    mb = 1000 if X_train.shape[0] > 1000 else None 
    
    
    # get the likelihood model (possibly a robust one)
    if gamma is None and beta is None:
        #standard likelihood
        lklh = Gaussian()
    elif beta is not None and gamma is None:
        #beta-divergence robustified likelihood
        lklh = betaDivGaussian(beta)
    elif gamma is not None and beta is None:
        #gamma-divergeece robustified likelihood
        lklh = gammaDivGaussian(gamma)
    else:
        print("ERROR! You have specified both beta and gamma. Either specify " + 
              "both as None (for standard Gaussian likelihood) or one of them " +
              "as None (to use the other)")
        sys.exit()
        
    
    """STEP (2): Call 'DGP' for split i, which together with ADAM is 
                 responsible for the inference"""
    model = DGP(X_train, Y_train, Z, kernels, lklh, #Gaussian(), #betaDivGaussian(0.01), #Gaussian(), #betaDivGaussian(0.1), #Gaussian(), #Gaussian_(), #gammaDivGaussian(0.1), #Gaussian_(), #gammaDivGaussian(0.01), #gammaDivGaussian(0.1), #Gaussian(), 
                num_samples=K, minibatch_size=mb, 
                alpha=alpha, white=white, div_weights=div_weights)

    # start the inner layers almost deterministically 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5
    
    
    #Build functions for evaluation test errors
    S = 100
    def batch_assess(model, assess_model, X, Y):
        n_batches = max(int(X.shape[0]/1000.), 1)
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
            l, sq = assess_model(model, X_batch, Y_batch)
            lik.append(l)
            sq_diff.append(sq)
        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
        return np.average(lik), np.average(sq_diff)**0.5
    
    def assess_single_layer(model, X_batch, Y_batch):
        m, v = model.predict_y(X_batch)
        lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5),  1)
        sq_diff = Y_std**2*((m - Y_batch)**2)
        return lik, sq_diff 
    
    def assess_sampled(model, X_batch, Y_batch):
        m, v = model.predict_y(X_batch, S)
        S_lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5), 2)
        lik = logsumexp(S_lik, 0, b=1/float(S))
        
        mean = np.average(m, 0)
        sq_diff = Y_std**2*((mean - Y_batch)**2)
        return lik, sq_diff

    #Get start time
    start_time = time.time()
    
    #Fit to training set via ADAM
    np.random.seed(1)
    AdamOptimizer(learning_rate).minimize(model, maxiter=iterations)
    
    #get running time
    running_time = time.time() - start_time
    s = 'time: {:.4f},  lik: {:.4f}, rmse: {:.4f}'
    
    """STEP (3): Extract and return test performancee metrics to 'main'."""
    #Get test errors
    lik, rmse = batch_assess(model, assess_sampled, X_test, Y_test)
    print(s.format(running_time, lik, rmse))

    return -lik, rmse, running_time




"""Function serves twp purposes: 
    (1) Calls 'get_test_error' for each split, which will build the DGP model
        for that split and return the test errors for the 10% hold out data
    (2) Store the results on the hard drive.
"""
def main(dataset, K, L, alpha, split_num, learning_rate, iterations, white,
         normalized, num_inducing, beta, gamma, robustness_mode,
         div_weights):

        
    #You can specify a selection of indices for the data splits, a single split
    #or None (in which case all splits are run)
    if split_num is None:
        n_splits = 50
        splits = range(n_splits)
    elif type(split_num) == int:
        splits = np.array([split_num])
    elif type(split_num) == list:
        splits = np.array(split_num)
    
    savepath =  '/results/'


    """STEP (1): Call 'get_test_error' for each split, building DGP model"""
    for i in splits: 
        print('split', i)
        
        if robustness_mode in ['g', 'b']:
            robustness_param = [e for e in [beta, gamma] if e is not None][0]
            file_string = (savepath + dataset + "_alpha" + str(alpha) + 
                           "_weights" + str(div_weights) + 
                           "_layers" + str(L) +
                            "_lr" + str(learning_rate) + "_iterations" + str(iterations) +
                            "_losstype_" + robustness_mode + str(robustness_param) +                   
                            "_split" + str(split_num))
        else:
            file_string = (savepath + dataset + "_alpha" + str(alpha) + 
                           "_weights" + str(div_weights) + 
                           "_layers" + str(L) +
                            "_lr" + str(learning_rate) + 
                            "_iterations" + str(iterations) +
                            "_split" + str(split_num))
        print(file_string)

        neg_test_ll, test_error, running_time = get_test_error(i, dataset, 
                    alpha, learning_rate, iterations, white, 
                    normalized, num_inducing, beta, gamma, div_weights)

        """STEP (2): Store the results on the hard drive"""
    
        with open(file_string + "_test_ll.txt", 'a') as f:
                f.write(repr(neg_test_ll) + '\n')
        with open(file_string + "_test_error.txt", 'a') as f:
                f.write(repr(test_error) + '\n')
        with open(file_string + "_test_time.txt", 'a') as f:
                f.write(repr(running_time) + '\n')



if __name__ == '__main__':
    """Refer to the README file to navigate & run the desired settings. 
    
    Important notes:
        - Since Renyi's alpha-divergence passes through the Kullback-Leibler 
            divergence for a, we have written this implementation i.t.o. 
            the alpha-renyi exclusively. To have the l-th layer KLD-regularized,
            simply specify alpha = 1.0 in the relevant entry of 'alphas_raw'
        - The number of layers L will be inferred from your input for the alphas 
            and divergence weights (div_weights). If the two objects don't
            match in length, execution is aborted.
    
    """
    
    
    
    """BLOCK I: Data & inference method"""
    dataset = str(sys.argv[1]) #boston, concrete, energy, kin8mn, naval, power,
                               #protein, wine, yacht
                                               
    #Convert raw input of form 1.0,1.0,0.5 into a list. The list holds the 
    #alpha-parameter of each layer-specific alpha-renyi divergence. 
    #Note: For alpha = 1.0, this recovers the KLD    
    alphas_raw = sys.argv[2]  
    alphas = [float(x) for x in alphas_raw.split(',')] 
                                                       
    #Convert raw input of form 1.0,1.0,2.0 into a list. The list holds the 
    #weights of each layer-specific divergence.                                       
    div_weights_raw = sys.argv[3] 
    div_weights = [float(x) for x in div_weights_raw.split(',')] 
    
    losstype = str(sys.argv[4])     #Type of loss used for (log-lklh, beta, gamma)
                                    #admissible choices: 
                                    #   'b' for beta-loss
                                    #   'g' for beta-loss
                                    #   'standard' negative log-loss
                                    
    loss_hyperparam = float(sys.argv[5])  #hyperparameter of the loss
                                          #beta if losstype = 'b'
                                          #gamma if losstype = 'g'
                                          #anything/nothing if losstype = 'standard'
                                          
                                          
#    dataset = 'boston'
#    L = 3
#    alphas = [0.5, 0.5, 0.5] #, 0.5, 0.5]
#    div_weights = [1.0, 1.0, 1.0] #, 1.0, 1.0]
#    learning_rate = 0.001
#    iterations = 20000
#    K = 5
#    num_inducing = 100
#    split_num = 0
#    losstype = 'b'
#    loss_hyperparam = 1.01
    
    """BLOCK II: Optimization hyperparameters (kept fixed in the paper) + 
                  some additional options for convenience."""
    
    #NOTE: All results obtained in the paper use the following settings:
    K = 5 #number of samples to approx E_q[l_n]
    num_inducing = 100 #Number of inducing points U
    split_num = None #could also be a list of splits/an single index (integer)
    iterations = 20000 #How many gradient steps we take
    white = True #Whitened GP representation inside gpflow
    normalized = True #Data normalization
    learning_rate = 0.01 #ADAM optimizer's default value.
                                    
                  
    """Sanity check I: Is div_weights of the same length as alphas? Set L
                       if so and abort otherwise."""
                       
    print('\n')                   
    if len(div_weights) == len(alphas):
        L = len(div_weights) #number of layers
    else:
        print("ERROR! number of divergence weights != number of alphas, " + 
              "but they have to match!")
        print("length of alphas: ", len(alphas))
        print("length of divergence weights: ", len(div_weights))
        print("Execution was aborted.")
        sys.exit(0)
        
    
    """Sanity check II: Did the user specify some alpha not in (0,1)?"""
    for alpha in alphas:
        if alpha > 1.0 or alpha <= 0.0:
            print("ERROR! You specified an invalid divergence for at least one " + 
                  "of the layers.")
            print("Only alpha in (0,1) can be guaranteed mathematically to " +
                  "be processed correctly! (see Thm. 6)")
            print("Execution was aborted.")
            sys.exit(0)
    
    """Sanity check III: Did the user specify some weight < 0?"""
    for weight in div_weights:
        if weight <= 0.0:
            print("ERROR! You specified an invalid divergence for at least one " + 
                  "of the layers.")
            print("Divergences need to be non-negative and have a unique minimum, " +
                  "so only weights >= 0 produce valid divergences!")
            print("Execution was aborted.")
            sys.exit(0)
    
    """Sanity check IV: If no robust loss specified, but robust loss hyperparam
                        is given, make sure to set hyperparam to -99 and inform
                        the user."""
    if losstype != 'g' and losstype != 'b' and loss_hyperparam != -99:
        print("WARNING! You specified a loss hyperparameter != -99, but " +
              "are using the standard log likelihood loss. " + 
              "The program forced your loss parameter to be -99!")
        print('\n')
        
        
    """Recasting: Convert the loss hyperparameter into the correct object & 
                  set all alpha = 1.0 too alpha = None for the internal 
                  processing of the DGP."""
    
    gamma, beta = None, None
    if losstype == 'b':
        #NOTE: We use a different parameterization for the code than inside the paper.
        #       In particular, beta_code = beta_paper - 1. 
        beta =  loss_hyperparam - 1.0 
    elif losstype == 'g':
        #NOTE: We use a different parameterization for the code than inside the paper.
        #       In particular, gamma_code = gamma_paper - 1. 
        gamma = loss_hyperparam - 1.0
    elif losstype != 'standard':
        print("WARNING! You specified an invalid losstype! " +
              "The program will use the standard log likelihood loss now!")
        print('\n')
        losstype = 'standard'
    
    
    #if alpha = 1.0, set alpha to None
    for i in range(0, len(alphas)):
        if alphas[i] == 1.0:
            alphas[i] = None
    
   
    #Q: Should I put Y and X in heree (or only X?)
    main(dataset, K, L, alphas, split_num, learning_rate, iterations, white,
         normalized, num_inducing, beta, gamma, losstype, div_weights)

