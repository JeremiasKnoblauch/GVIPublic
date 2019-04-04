#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:16:25 2019

@author: Jeremias Knoblauch

Description: build robust likelihoods using the beta and gamma div
             builds upon gpflow and can be used within the DSDGP code
"""

import tensorflow as tf
from gpflow.likelihoods import Gaussian
from gpflow.decors import params_as_tensors
import numpy as np


class betaDivGaussian(Gaussian):
    """Extends the Gaussian likelihood. Note that virtually everything is
    unaffected except for the LEARNING, i.e. the loss that is produced inside
    the expectation over q. We still predict from the same model as before, 
    so nothing changes in any of the other functionality"""
    
    def __init__(self, beta = 0.01):
        super().__init__() #Simply initialize like you would a Gaussian lklh.
        self.beta = beta #additionally, we add the robustifier in here.
                         #Note that we use a different parameterization from
                         #the paper: beta_paper = beta_code + 1
    
    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        """This gives you the beta-divergence losses evaluated for the sample
        Y drawn from q. In particular, this means we return for beta = b
        
            -(1/b) * N(Y|Fmu, Fvar)^{b} + (1/(b+1)) * Integral(x; N(x|Fmu,Fvar)^(b+1))
        
        Note: Fmean, Fvar here are samples from the variational posterior
        Note: Notation from paper appendix: Fmu = mu, Fvar = Sigma
        Note: We integrate out the beta-losses in closed form as in the Appendix
        Note: We copute the two terms separately: The first one in log form, the
              second one without log form. We then re-combine them separately
              later on.
        """
     
        Sigma_tilde = 1.0 / (self.beta  / self.variance + 1.0 / Fvar ) 
        mu_tilde = ((self.beta / self.variance) * Y  + Fmu / Fvar)
        log_integral = (-0.5*self.beta) * tf.log(2.0 * np.pi * self.variance) \
                         -0.5 * np.log(1 + self.beta)
        log_Sigma_tilde = -tf.log(self.beta / self.variance + 1.0 / Fvar )
        
        log_tempered = (-np.log(self.beta) + 
            (-0.5 * self.beta) * (tf.log(self.variance * 2.0 * np.pi)) +
            0.5 * (log_Sigma_tilde - tf.log(Fvar)) +
            (-0.5 * ( 
                    (self.beta * (Y ** 2) / self.variance)  + 
                    ((Fmu ** 2) / Fvar) - 
                    ((mu_tilde ** 2) * Sigma_tilde )))  
        )

        return log_tempered, log_integral


class gammaDivGaussian(Gaussian):
    """Extends the Gaussian likelihood. Note that virtually everything is
    unaffected except for the LEARNING, i.e. the loss that is produced inside
    the expectation over q. We still predict from the same model as before, 
    so nothing changes in any of the other functionality"""
    
    def __init__(self, gamma = 0.01):
        super().__init__() #Simply initialize like you would a Gaussian lklh.
        self.gamma = gamma #additionally, we add the robustifier in here.
                         #Note that we use a different parameterization from
                         #the paper: gamma_paper = gamma_code + 1
    
    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        """This gives you the beta-divergence losses evaluated for the sample
        Y drawn from q. In particular, this means we return for gamma = g
        
            -(1/g) * N(Y|Fmu, Fvar)^{g} * [(1/(b+1))Integral(x; N(x|Fmu,Fvar)^(g+1))^(-g/(1+g))]
        
        Note: Fmean, Fvar here are samples from the variational posterior
        Note: Notation from paper appendix: Fmu = mu, Fvar = Sigma
        Note: We integrate out the beta-losses in closed form as in the Appendix
        Note: Unlike for the beta-div loss, we can compute everything in log
                form for the gamma loss & only need to return a single 
                quantity.
        """
        
        Sigma_tilde = 1.0 / (self.gamma  / self.variance + 1.0 / Fvar ) 
        mu_tilde = ((self.gamma / self.variance) * Y  + Fmu / Fvar)
        log_integral = (-0.5*self.gamma) * tf.log(2.0 * np.pi * self.variance) \
                         -0.5 * np.log(1 + self.gamma)
        log_Sigma_tilde = -tf.log(self.gamma / self.variance + 1.0 / Fvar )
        
        log_tempered = (-np.log(self.gamma) + 
            (-0.5 * self.gamma) * (tf.log(self.variance * 2.0 * np.pi)) +
            0.5 * (log_Sigma_tilde - tf.log(Fvar)) +
            (-0.5 * ( 
                    (self.gamma * (Y ** 2) / self.variance)  + 
                    ((Fmu ** 2) / Fvar) - 
                    ((mu_tilde ** 2) * Sigma_tilde )))  
        )

                
        #S, N, D     
        return (tf.cast(log_tempered - 
                         (-self.gamma/(1+self.gamma))*log_integral + 
                         (1+self.gamma), 
                         dtype = tf.float64))