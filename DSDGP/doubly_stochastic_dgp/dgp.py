# This is a carbon copy of the class with same name corresponding to 
# Salimbeni & Deisenroth '17. 
#
# I have enhanced this to deal with 
#       * a weight w_l on the l-th layer
#           -See lines 61-77
#       * the different log/no-log-representations of the beta & gamma losses
#           -See lines 134-152, 157-165

import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import settings
float_type = settings.float_type

from robustified_likelihoods import betaDivGaussian, gammaDivGaussian

from doubly_stochastic_dgp.utils import reparameterize

from doubly_stochastic_dgp.utils import BroadcastingLikelihood
from doubly_stochastic_dgp.layer_initializations import init_layers_linear
from doubly_stochastic_dgp.layers import GPR_Layer, SGPMC_Layer, GPMC_Layer, SVGP_Layer


class DGP_Base(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=1, num_data=None,
                 div_weights = None,
                 **kwargs):
        Model.__init__(self, **kwargs)
        self.num_samples = num_samples

        self.num_data = num_data or X.shape[0]
        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = BroadcastingLikelihood(likelihood)

        self.layers = ParamList(layers)
        
        """CHANGES START"""
        """Weights for the uncertainty quantifiers (per layer)"""
        if div_weights is None:
            div_weights = [1.0] * len(layers) #multiply by 1, i.e. don't change
        elif type(div_weights) == list and len(div_weights) != len(layers):
            print("WARNING! You specified a list of weights for the " + 
                  "uncertainty quantifiers, but your DGP has more/less layers " +
                  "than the number of weights you specified! " +
                  "We set all weights to 1.0" )
            div_weights = [1.0] * len(layers)
        elif type(div_weights) == list and len(div_weights) == len(layers):
            div_weights = div_weights
            
        """Distribute the weights into the layers"""
        for layer, weight in zip(layers, div_weights):
            layer.set_weight(weight)
        """CHANGES EEND"""
        

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """

        Fmean, Fvar = self._build_predict(X, full_cov=False, S=self.num_samples)
        
        """
        Below function:
             
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        
        """CHANGES START"""
        # convert from S, N, D => N, D 
        if (isinstance(self.likelihood.likelihood,gammaDivGaussian)):
           #For the gamma-div, the loss is strictly > 0 so we can log everything 
           return tf.reduce_logsumexp(
                tf.cast(self.likelihood.variational_expectations(Fmean, Fvar, Y), dtype=tf.float64)
                , 0)        - np.log(self.num_samples)
        elif (isinstance(self.likelihood.likelihood,betaDivGaussian)):
            #For the beta-div, we need to treat both terms separately
            log_tempered, log_integral = self.likelihood.variational_expectations(Fmean, Fvar, Y)
            log_tempered_avg = tf.reduce_logsumexp(tf.cast(log_tempered, dtype=tf.float64)
                , 0) - np.log(self.num_samples)
            return log_tempered_avg, log_integral
            
        else:
            #Standard procedure of original code
            var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
            return tf.reduce_mean(var_exp, 0)  # N, D
        """CHANGES END"""

    @params_as_tensors
    def _build_likelihood(self):
        
        """CHANGES START"""
        if isinstance(self.likelihood.likelihood,gammaDivGaussian):
            L = tf.exp(tf.reduce_logsumexp(self.E_log_p_Y(self.X, self.Y)) )
        elif isinstance(self.likelihood.likelihood,betaDivGaussian):
            L1, L2 = self.E_log_p_Y(self.X, self.Y)
            L =  tf.exp(tf.reduce_logsumexp(L1)) + self.num_data * tf.cast(tf.shape(self.X)[0], float_type)
        else:
            L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        """CHANGES END"""
        
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)


class DGP_Quad(DGP_Base):
    """
    A DGP with quadrature instead of MC sampling. This scales exponentially in the sum of the inner layer dims
    
    The key ref is:
    [in progress]
    
    """
    def __init__(self, *args, H=100, **kwargs):
        DGP_Base.__init__(self, *args, **kwargs)

        # set up the quadrature points
        self.H = H
        self.D_quad = sum([layer.q_mu.shape[1] for layer in self.layers[:-1]])
        gh_x, gh_w = mvhermgauss(H, self.D_quad)
        gh_x *= 2. ** 0.5  # H**quad_dims, quad_dims
        self.gh_w = gh_w * np.pi ** (-0.5 * self.D_quad)  # H**quad_dims

        # split z into each layer, to work with the loop over layers
        # the shape is S, 1, D as this will broadcast correctly with S,N,D (never used with full cov)
        s, e = 0, 0
        self.gh_x = []
        for layer in self.layers[:-1]:
            e += layer.q_mu.shape[1]
            self.gh_x.append(gh_x[:, None, s:e])
            s += layer.q_mu.shape[1]

        # finish with zeros (we don't need to do quadrature over the final layer and this will never get used
        self.gh_x.append(tf.zeros((1, 1, 1), dtype=settings.float_type))

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with quadrature 
        """
        _, Fmeans, Fvars = self.propagate(X, zs=self.gh_x, full_cov=False, S=self.H**self.D_quad)
        var_exp = self.likelihood.variational_expectations(Fmeans[-1], Fvars[-1], Y)  # S, N, D
        return tf.reduce_sum(var_exp * self.gh_w[:, None, None], 0)  # N, D


class DGP(DGP_Base):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, likelihood,
                 num_outputs=None,
                 mean_function=Zero(),  # the final layer mean function,
                 white=False, 
                 alpha = None, #If we want alpha-renyi 
                 **kwargs):
        layers = init_layers_linear(X, Y, Z, kernels,
                                    num_outputs=num_outputs,
                                    mean_function=mean_function,
                                    white=white, 
                                    alpha=alpha)
        DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)

