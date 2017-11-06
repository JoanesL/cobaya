"""
.. module:: samplers.mcmc

:Synopsis: Basic, inefficient ellipsoidal nested sampler. For tests -- better use PolyChord instead.
:Author: 

"""

# Python 2/3 compatibility
from __future__ import division

# Global
from copy import deepcopy
from itertools import chain
import numpy as np

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi, get_mpi_size, get_mpi_rank, get_mpi_comm
from cobaya.collection import Collection, OnePoint
from cobaya.conventions import _weight, _p_proposal, _minuslogpost
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.log import HandledException
from cobaya.tools import get_external_function

# Logger
import logging
log = logging.getLogger(__name__)


        # nlive_dim: 20
        # enlargement_factor: 2
        # evidence_tolerance: 0.01
        # ntotal_dim: inf



class nested(Sampler):

    def initialise(self):
        self.nlive = self.prior.d() * self.nlive_dim
        self.live = Collection(
            self.parametrisation, self.likelihood, self.output, name="live",
            initial_size=self.nlive)
        self.dead = Collection(
            self.parametrisation, self.likelihood, self.output, name="sample")
# OLD collections
#        self.live_x = np.zeros((self.nlive, prior.dimension()))
#        self.live_loglik = np.zeros(self.nlive)
#        self.samples_x = np.zeros((self.nlive, prior.dimension()))
#        self.samples_loglik = np.zeros(self.nlive)
#        self.samples_weight = np.zeros(self.nlive)
        self.logZ = -np.inf
        # Keeping track of stuff for efficiency (1-based)
        self.logZ_i = []
        self.logZ_sigma_i = []
        self.n_eval_i = [0]
        # Ending conditions
        self.evidence_tolerance_i = 0
        log.info("Initialised")

        self.sample_from_isocontour = self.sample_from_isocontour_bruteforce

    def run(self):
        log.info("Generating initial live sample ...")
        while self.live.n() < self.nlive:
            sample = self.prior.sample(external_error=False)[0]
            logpost, logprior, logliks, derived = self.logposterior(sample)
            self.n_eval_i[0] += 1
            if logpost > -np.inf:
                self.live.add(sample, derived=derived, logpost=logpost,
                              logprior=logprior, logliks=logliks)
        log.info("Done generating live samples.")
        self.iteration = 0
        while not self.has_converged():
            self.iteration += 1
            self.n_eval_i += [0]
            log.info("Iteration #%d -----------------------------------", self.iteration)
#            exit()
            i_min_logp = self.live.data[_minuslogpost].idxmax()
            self.live.add_i_to_collection(i_min_logp, self.dead)
            min_logp = -self.live[_minuslogpost][i_min_logp]
            log_ring_width = -self.iteration/self.nlive + np.log(np.exp(1/self.nlive)-1)
            print min_logp, log_ring_width, "======================="
                                                            
            Z_increment = np.exp(min_logp+log_ring_width)
            self.dead.data.set_value(_weight, -1, Z_increment)
            self.logZ = np.log(np.exp(self.logZ)+Z_increment)
            self.logZ_i += [self.logZ]
            sample, logpost, logprior, logliks, derived = (
                self.sample_from_isocontour(min_logp))
            self.live.add(sample, derived=derived, logpost=logpost,
                          logprior=logprior, logliks=logliks, at=i_min_logp)
            log.info("Sampling efficiency: %.3f (now) , %.3f (total)",
                     1/self.n_eval_i[self.iteration],
                     self.iteration/sum(self.n_eval_i[1:]))
            log.info("New dead sample: %r", self.dead.data.iloc[self.iteration])
            log.info("New live sample: %r", self.live.data.iloc[i_min_logp])

            
# COMPROBAR QUE NO ME HE CARGADO LA COLLECTION!!!!!
                     
# CREO QUE COMO AQU'I S'I SAMPLEO DEL PRIOR, NO TENGO QUE INTEGRAR EL POST
# SINO LA LIK (Y QUIZA LOS PRIORS EXTERNOS --> COMPROBARLO!!!!

# anhadir a add_to_collection la seleccion de que punto pasar:
# collection_instance.add_to_collection(destiny_collection, n=x)

    def has_converged(self):
        log.info("log-evidence and uncert: %g +/- ?????", self.logZ)
#         vol = np.exp(-self.i_iter/float(self.nlive))
#         log_expected_contribution = np.log(self.live_loglik.max()*vol)
#         if log_expected_contribution <= self.logZ+np.log(self.evidence_tolerance):
#             self.evidence_tolerance_i += 1
#             if self.evidence_tolerance_i >= self.evidence_tolerance_times:
#                 self.myprint("Finished: evidence tolerance achieved %d times in a row."%(
#                     self.evidence_tolerance_times))
#                 self.print_final_result()
#                 return True
#         else:
#             self.evidence_tolerance_i = 0
#         if self.i_iter >= self.n_total:
#             self.myprint("Finished: number of iterations > total allowed.")
#             self.print_final_result()
#             return True
#         return False

    def sample_from_isocontour_bruteforce(self, min_logp):
        while True:
            sample = self.prior.sample(external_error=False)[0]
            logpost, logprior, logliks, derived = self.logposterior(sample)
            self.n_eval_i[self.iteration] += 1
            if logpost > min_logp:
                return sample, logpost, logprior, logliks, derived

#     def sample_from_isocontour_ellipsoid(self):
#         # bounds not used for now: lims = self.box_approximation()
#         if self.prior.dimension() == 1:
#             lims = self.box_approximation()
#             centre = np.array([(lims[0,1]+lims[0,0])/2.])
#             semiaxes = np.array([(lims[0,1]-lims[0,0])/2.])
#             rotation = np.array([[1]])
#         else:
#             centre, semiaxes, rotation = self.ellipsoid_approximation()
#         while True:
#             x = self.prior.sample()
#             while self.ellipsoid_condition(x, centre, semiaxes, rotation) > 1:
#                 x = self.prior.sample() #(bounds=lims)
#             loglik = self.likelihood.loglik(x)
#             self.n_lik_eval_iter[self.i_iter] += 1
#             if loglik > self.minloglik:
#                 return x, loglik

#     def box_approximation(self):
#         lims = np.array([[self.live_x[:,i].min(), self.live_x[:,i].max()]
#                          for i in range(self.prior.dimension())])
#         # Exact enlargement of the minimum box given by the enlargement factor: 
#         increase = (lims[:,1]-lims[:,0])*self.enlargement_factor/4.
#         # Additional enlargement to contain the enlarged ellipse if it is highly deg.:
#         increase *= self.enlargement_factor
#         lims[:,0] -= increase
#         lims[:,1] += increase
#         return lims

#     def box_projection(self, i, j, lims):
#         return Rectangle((lims[i,0],lims[j,0]), width=lims[i,1]-lims[i,0],
#                           height=lims[j,1]-lims[j,0], facecolor="none")
        
#     def ellipsoid_approximation(self):
#         """
#         Computes the minimum engulfing ellipsoid.
#         Based on section 3.2 of `Mukherjee, Parkinson Liddle 2006 <http://iopscience.iop.org/article/10.1086/501068/pdf>`_. 
#         See also section 3.2 of Sivia & Skilling 2006, *Data Analysis: a Bayesian tutorial*.
#         """
#         if self.prior.dimension() == 1:
#             lims = self.box_approximation()
#             centre = np.array([(lims[0,1]+lims[0,0])/2.])
#             semiaxes = np.array([(lims[0,1]-lims[0,0])/2.])
#             rotation = np.array([[1]])
#         else:
#             centre = np.mean(self.live_x, axis=0)
#             eigenv, rotation = np.linalg.eigh(np.cov(self.live_x, rowvar=0))
#             semiaxes = np.power(eigenv, 0.5)
#             # 2-sigma contour: *2
#             semiaxes *=2
#             semiaxes *= self.enlargement_factor
#         return centre, semiaxes, rotation

#     def ellipsoid_condition(self, x, centre, semiaxes, rotation):
#         return (((x-centre).dot(rotation)/semiaxes)**2).sum(axis=-1)
    
#     def ellipsoid_projection(self, i, j, centre, semiaxes, rotation):
#         assert self.prior.dimension() == 2, "Not implemented for !=2d, sorry!"
#         return Ellipse(centre, 2*semiaxes[0], 2*semiaxes[1],
#                        -np.degrees(np.arccos(rotation[0,0])), facecolor="none")
