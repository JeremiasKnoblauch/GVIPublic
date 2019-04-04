# This is a carbon copy of the class with same name corresponding to 
# Salimbeni & Deisenroth '17. 
#
# I have enhanced the function 'KL' to deal with the renyi-alpha regularizer
# to the prior and added a class attribute + setter function, too. 
# See lines 40, 227 - 353, 117-118.
#
# I have also added the layer-specific divergence weight & setter fct, see lines 
# 40, 120/21, 265, 355

import tensorflow as tf
import numpy as np

from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
from gpflow.kullback_leiblers import gauss_kl
from gpflow.priors import Gaussian as Gaussian_prior
from gpflow import transforms
from gpflow import settings
from gpflow.models.gplvm import BayesianGPLVM
from gpflow.expectations import expectation
from gpflow.probability_distributions import DiagonalGaussian
from gpflow import params_as_tensors
from gpflow.densities import multivariate_normal



from doubly_stochastic_dgp.utils import reparameterize


class Layer(Parameterized):
    def __init__(self, input_prop_dim=None, **kwargs):
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        :param input_prop_dim: the first dimensions of X to propagate. If None (or zero) then no input prop
        :param kwargs:
        """
        Parameterized.__init__(self, **kwargs)
        self.input_prop_dim = input_prop_dim
        self.alpha = None
        self.weight = 1.0

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = self.num_outputs

        mean = tf.reshape(mean, (S, N, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, N, D))

        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = tf.concat([X_prop, samples], 2)
            mean = tf.concat([X_prop, mean], 2)

            if full_cov:
                shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[1], tf.shape(var)[3])
                zeros = tf.zeros(shape, dtype=settings.float_type)
                var = tf.concat([zeros, var], 3)
            else:
                var = tf.concat([tf.zeros_like(X_prop), var], 2)

        return samples, mean, var
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_weight(self, weight):
        self.weight = weight


class SVGP_Layer(Layer):
    def __init__(self, kern, Z, num_outputs, mean_function,
                 white=False, input_prop_dim=None, **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :return:
        """
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.num_inducing = Z.shape[0]

        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu)

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)

        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs
        self.white = white #tf.constant(white, shape=(), dtype = tf.bool)        #white #

        if not self.white:  # initialize to prior
            Ku = self.kern.compute_K_symm(Z)
            Lu = np.linalg.cholesky(Ku + np.eye(Z.shape[0])*settings.jitter)
            self.q_sqrt = np.tile(Lu[None, :, :], [num_outputs, 1, 1])

        self.needs_build_cholesky = True

    @params_as_tensors
    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            self.Ku = self.feature.Kuu(self.kern, jitter=settings.jitter)
            self.Lu = tf.cholesky(self.Ku)
            self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
            self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
            #also compute K_inverse and it's det
            if not self.white:
                inp_ = (self.Ku + 
                        tf.eye(self.num_inducing, dtype=tf.float64)*settings.jitter*10)
                self.K_inv = tf.linalg.inv(tf.cast(inp_, dtype=tf.float64))
            
            self.needs_build_cholesky = False


    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        # mmean, vvar = conditional(X, self.feature.Z, self.kern,
        #             self.q_mu, q_sqrt=self.q_sqrt,
        #             full_cov=full_cov, white=self.white)
        Kuf = self.feature.Kuf(self.kern, X)

        A = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True)
        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)

        mean = tf.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)


        B = tf.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kern.K(X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = self.kern.Kdiag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        return mean + self.mean_function(X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior. 
        Notation in paper is KL[q(u)||p(u)]. 
        
        OR the alpha-renyi divergence from variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt * q_sqrt^T) to N(0, I) (if whitened)
                and to N(mu(Z), K(Z)) otherwise, independently for each GP
        """
        # if self.white:
        #     return gauss_kl(self.q_mu, self.q_sqrt)
        # else:
        #     return gauss_kl(self.q_mu, self.q_sqrt, self.Ku)
        #

        self.build_cholesky_if_needed()
        if self.alpha is None:
            """Get KL regularizer"""

            KL = -0.5 * self.num_outputs * self.num_inducing
            KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))
    
            if not self.white:
                # Whitening is relative to the prior. Here, the prior is NOT 
                # whitened, meaning that we have N(0, K(Z,Z)) as prior.
                KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) * self.num_outputs
                KL += 0.5 * tf.reduce_sum(tf.square(
                        tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, 
                                                   lower=True)))
                Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)
                KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
            else:
                KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
                KL += 0.5 * tf.reduce_sum(self.q_mu**2)
    
            return self.weight * KL
        else:
            """Get AR regularizer. For the normal, this means 
            log(Normalizing Constant[alpha * eta_q + (1-alpha) * eta_0 ]) - 
            alpha*log(Normalizing Constant[eta_q]) - 
            (1-alpha)*log(Normalizing Constant[eta_0]).
            
            NOTE: the 2*pi factor will cancel, as well as the 0.5 * factor.
            NOTE: q_strt is s.t. q_sqrt * q_sqrt^T = variational variance, i.e.
                  q(v) = N(q_mu, q_sqrt q_sqrt^T).
            NOTE: self.Lu is cholesky decomp of self.Ku
            NOTE: self.feature are the inducing points Z, and 
                  self.Ku = self.feature.Kuu(kernel), meaning that self.Ku is
                  the kernel matrix computed at the inducing points Z.
            NOTE: We need the alpha-renyi div between prior and GP-variational
                  posterior for EACH of the GPs in this layer.
                  
            Shapes:
                q_sqrt:                                 13 x 100 x 100
                q_mu:                                   100 x 13
                tf.matrix_diag_part(self.q_sqrt):       13 x 100
                q_sqrt_inv:                             13 x 100 x 100
                Ku, Lu:                                 100 x 100
                num_inducing:                           100
                num_outputs:                            13
            """
            
            #convenience
            alpha = self.alpha
            
            #INEFFICIENT, can probably be done much better with cholesky solve
            inp_ = (tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)
                   + tf.eye(self.num_inducing, dtype=tf.float64)*settings.jitter*100)
            q_inv = tf.linalg.inv(tf.cast(inp_, dtype=tf.float64))
            
            #gives Sigma_q^-1 * mu_q
            q_var_x_q_mu = tf.matmul(q_inv, 
                tf.reshape(self.q_mu, shape = (self.num_outputs, self.num_inducing, 1)))
            
            #Get the two log-normalizers for the variational posteriors
            q_component_1 = 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))
            q_component_2 = 0.5 * tf.reduce_sum(q_var_x_q_mu * self.q_mu)
            
            logZq = (  q_component_1 + q_component_2)
            
                            
            if not self.white:
                #prior using self.Lu, still 0 mean fct
                logZpi = 0.5 * tf.reduce_sum(tf.log(
                        tf.matrix_diag_part(self.Lu) ** 2)) * self.num_outputs 
                
                new_Sigma_inv = (alpha * q_inv + (1.0 - alpha) * self.K_inv 
                     + tf.eye(self.num_inducing, dtype=tf.float64)*settings.jitter)# + 
                
                
            else:
                logZpi = 0.0 #* self.num_outputs * self.num_inducing  - but that is still 0.
                new_Sigma_inv = (alpha * q_inv + (1.0 - alpha + settings.jitter) * 
                                 tf.eye(self.num_inducing, dtype=tf.float64))
            
            
            new_Sigma_inv_chol = tf.cholesky( tf.cast(new_Sigma_inv, tf.float64))
            log_det = -tf.reduce_sum(tf.log(tf.matrix_diag_part(new_Sigma_inv_chol) ** 2))
            
            #Get the new inverse variance of the exponential family member 
            #corresponding to alpha * eta_q + (1-alpha) * eta_0.
            #var_inv_new = tf.matmul(chol_var_inv_new, chol_var_inv_new, transpose_b=True)
            
            #Compute mu_new: Compute (Sigma^-1*mu) = A via
            #A = alpha* Sigma_q^-1 * q_mu + (1-alpha * 0) and then multiply
            #both sides by Sigma! => Problem: I don't know sigma!
            mu_new = tf.linalg.solve(tf.cast(new_Sigma_inv, dtype = tf.float64),
                                       tf.cast(alpha * q_var_x_q_mu, 
                                               dtype = tf.float64))
            
            #Note: Sigma^{-1}_new * mu_new = Sigma^{-1}_q * mu_q, so 
            #      mu_new' * Sigma^{-1}_new * mu_new = mu_new' * (Sigma^{-1}_q * mu_q)
            mu_new_x_new_Sigma_inv = tf.reduce_sum(alpha * q_var_x_q_mu * mu_new)
            
            #Observing that log(|Sigma|) = - log(|Sigma|^-1), we can now get 
            #the normalizing constant of the new exp. fam member.
            logZnew = (0.5 * mu_new_x_new_Sigma_inv + 
                0.5 * log_det
                )
            
            #return the log of the AR-div between the normals, i.e. 
            # (1/(alpha * (1-alpha))) * log(D), where D = 
            #new normalizer / (q_normalizer^alpha * prior_normalizer^(1-alpha))
            AR = (1.0/(alpha * (1.0-alpha))) * (logZnew - alpha * logZq - (1.0-alpha) * logZpi)
            
            return self.weight * AR


class SGPMC_Layer(SVGP_Layer):
    def __init__(self, *args, **kwargs):
        """
        A sparse layer for sampling over the inducing point values 
        """
        SVGP_Layer.__init__(self, *args, **kwargs)
        self.q_mu.prior = Gaussian_prior(0., 1.)
        del self.q_sqrt
        self.q_sqrt = None

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)


class GPMC_Layer(Layer):
    def __init__(self, kern, X, num_outputs, mean_function, input_prop_dim=None, **kwargs):
        """
        A dense layer with fixed inputs. NB X does not change here, and must be the inputs. Minibatches not possible
        """
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.num_data = X.shape[0]
        q_mu = np.zeros((self.num_data, num_outputs))
        self.q_mu = Parameter(q_mu)
        self.q_mu.prior = Gaussian_prior(0., 1.)
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs

        Ku = self.kern.compute_K_symm(X) + np.eye(self.num_data) * settings.jitter
        self.Lu = tf.constant(np.linalg.cholesky(Ku))
        self.X = tf.constant(X)

    def build_latents(self):
        f = tf.matmul(self.Lu, self.q_mu)
        f += self.mean_function(self.X)
        if self.input_prop_dim:
            f = tf.concat([self.X[:, :self.input_prop_dim], f], 1)
        return f

    def conditional_ND(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              full_cov=full_cov,
                              q_sqrt=None, white=True)
        return mu + self.mean_function(Xnew), var


class Collapsed_Layer(Layer):
    """
    Extra functions for a collapsed layer
    """
    def set_data(self, X_mean, X_var, Y, lik_variance):
        self._X_mean = X_mean
        self._X_var = X_var
        self._Y = Y
        self._lik_variance = lik_variance

    def build_likelihood(self):
        raise NotImplementedError


class GPR_Layer(Collapsed_Layer):
    def __init__(self, kern, mean_function, num_outputs, **kwargs):
        """
        A dense GP layer with a Gaussian likelihood, where the GP is integrated out
        """
        Collapsed_Layer.__init__(self, **kwargs)
        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs

    def conditional_ND(self, Xnew, full_cov=False):
        ## modified from GPR
        Kx = self.kern.K(self._X_mean, Xnew)
        K = self.kern.K(self._X_mean) + tf.eye(tf.shape(self._X_mean)[0], dtype=settings.float_type) * self._lik_variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self._Y - self.mean_function(self._X_mean))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self._Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self._Y)[1]])
        return fmean, fvar

    def build_likelihood(self):
        ## modified from GPR
        K = self.kern.K(self._X_mean) + tf.eye(tf.shape(self._X_mean)[0], dtype=settings.float_type) * self._lik_variance
        L = tf.cholesky(K)
        m = self.mean_function(self._X_mean)
        return tf.reduce_sum(multivariate_normal(self._Y, m, L))


class SGPR_Layer(Collapsed_Layer):
    def __init__(self, kern, Z, num_outputs, mean_function, **kwargs):
        """
        A sparse variational GP layer with a Gaussian likelihood, where the 
        GP is integrated out

        :kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function
        :return:
        """

        Collapsed_Layer.__init__(self, **kwargs)
        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs

    def conditional_ND(self, Xnew, full_cov=False):
        return gplvm_build_predict(self, Xnew, self._X_mean, self._X_var, self._Y, self._lik_variance, full_cov=full_cov)

    def build_likelihood(self):
        return gplvm_build_likelihood(self, self._X_mean, self._X_var, self._Y, self._lik_variance)


################## From gpflow (with KL removed)
def gplvm_build_likelihood(self, X_mean, X_var, Y, variance):
    if X_var is None:
        # SGPR
        num_inducing = len(self.feature)
        num_data = tf.cast(tf.shape(Y)[0], settings.float_type)
        output_dim = tf.cast(tf.shape(Y)[1], settings.float_type)

        err = Y - self.mean_function(X_mean)
        Kdiag = self.kern.Kdiag(X_mean)
        Kuf = self.feature.Kuf(self.kern, X_mean)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = tf.cholesky(Kuu)
        sigma = tf.sqrt(variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.log(variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.matrix_diag_part(AAT))

        return bound


    else:

        X_cov = tf.matrix_diag(X_var)
        pX = DiagonalGaussian(X_mean, X_var)
        num_inducing = len(self.feature)
        if hasattr(self.kern, 'X_input_dim'):
            psi0 = tf.reduce_sum(self.kern.eKdiag(X_mean, X_cov))
            psi1 = self.kern.eKxz(self.feature.Z, X_mean, X_cov)
            psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.feature.Z, X_mean, X_cov), 0)
        else:
            psi0 = tf.reduce_sum(expectation(pX, self.kern))
            psi1 = expectation(pX, (self.kern, self.feature))
            psi2 = tf.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = tf.cholesky(Kuu)
        sigma2 = variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        # dX_var = self.X_var if len(self.X_var.get_shape()) == 2 else tf.matrix_diag_part(self.X_var)
        # NQ = tf.cast(tf.size(self.X_mean), settings.float_type)
        D = tf.cast(tf.shape(Y)[1], settings.float_type)
        # KL = -0.5 * tf.reduce_sum(tf.log(dX_var)) \
        #      + 0.5 * tf.reduce_sum(tf.log(self.X_prior_var)) \
        #      - 0.5 * NQ \
        #      + 0.5 * tf.reduce_sum((tf.square(self.X_mean - self.X_prior_mean) + dX_var) / self.X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(Y), settings.float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        # bound -= KL # don't need this term
        return bound

############# Exactly from gpflow
def gplvm_build_predict(self, Xnew, X_mean, X_var, Y, variance, full_cov=False):
    if X_var is None:
        # SGPR
        num_inducing = len(self.feature)
        err = Y - self.mean_function(X_mean)
        Kuf = self.feature.Kuf(self.kern, X_mean)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma = tf.sqrt(variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    else:
        # gplvm
        pX = DiagonalGaussian(X_mean, X_var)
        num_inducing = len(self.feature)

        X_cov = tf.matrix_diag(X_var)

        if hasattr(self.kern, 'X_input_dim'):
            psi1 = self.kern.eKxz(self.feature.Z, X_mean, X_cov)
            psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.feature.Z, X_mean, X_cov), 0)
        else:
            psi1 = expectation(pX, (self.kern, self.feature))
            psi2 = tf.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)

        # psi1 = expectation(pX, (self.kern, self.feature))
        # psi2 = tf.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)

        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma2 = variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
