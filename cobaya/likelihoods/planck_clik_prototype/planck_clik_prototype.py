"""
.. module:: planck_clik_prototype

:Synopsis: Definition of the clik code based likelihoods
:Author: Julien Lesgourgues and Benjamin Audren (and Jesus Torrado for small compatibility changes only)

Contains the definition of the base clik 2.0 likelihoods, from which all Planck 2015
likelihoods inherit.

The treatment on Planck 2015 likelihoods has been adapted without much modification from
the `MontePython <http://baudren.github.io/montepython.html>`_ code by Julien Lesgourgues
and Benjamin Audren.

.. note::
  
  An older, `MIT-licensed <https://opensource.org/licenses/MIT>`_ version of this module by
  Julien Lesgourgues and Benjamin Audren can be found in the `source of
  MontePython <https://github.com/baudren/montepython_public>`_.


The Planck 2015 likelihoods defined here are:

- ``planck_2015_lowl``
- ``planck_2015_lowTEB``
- ``planck_2015_plikHM_TT``
- ``planck_2015_plikHM_TTTEEE``
- ``planck_2015_plikHM_TTTEEE_unbinned``
- ``planck_2015_plikHM_TT_unbinned``


Usage
-----

To use the Planck likelihoods, you simply need to mention them in the likelihood blocks,
specifying the path where you have installed them, for example:

.. code-block:: yaml

   likelihood:
     planck_2015_lowTEB:
       path: /path/to/cosmo/likelihoods/planck2015/
     planck_2015_plikHM_TTTEEE:
       path: /path/to/cosmo/likelihoods/planck2015/

This automatically pulls the likelihood parameters into the sampling, with their default
priors. The parameters can be fixed or their priors modified by re-defining them in the
``params: likelihood:`` block.

The default parameter and priors can be found in the ``defaults.yaml`` files in the
folder corresponding to each likelihood. They are not reproduced here because of their
length.

.. [The installation instructions are in the corresponding doc file because they need
..  some unicode characters to show some directory structure]

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import numpy as np

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)

class planck_clik_prototype(Likelihood):

    def initialise(self):
        self.name = self.__class__.__name__
        # Importing Planck's clik library (only once!)
        if self.path == None:
            log.error("No path given to the Planck likelihood. Set the likelihood "
                      "property 'path' to the folder containing 'plc-2.0'.")
            raise HandledException
        try:
            clik
        except NameError:
            try:
                clik_path = os.path.join(self.path, "plc-2.0")
                log.info("[%s] Importing clik from %s", self.name, clik_path)
                if not os.path.exists(clik_path):
                    log.error("The given folder does not exist: '%s'",clik_path)
                    raise HandledException
                clik_path = os.path.join(clik_path, "lib/python2.7/site-packages")
                if not os.path.exists(clik_path):
                    log.error("You have not compiled the Planck likelihood code 'clik'.\n"
                              "Take a look at the docs to see how to do it using 'waf'.")
                    raise HandledException
                sys.path.insert(0, clik_path)
                import clik
            except ImportError:
                log.error("Could not import the Planck likelihood. "
                          "This shouldn't be happening. Contact the developer.")
                raise HandledException
        # Loading the likelihood data
        if self.path is None:
            log.error("You must provide a path to the Planck likelihood, "
                      "up to the location of the folders 'low_l', 'hi_l', etc.")
            raise HandledException
        clik_file = os.path.join(self.path,"plc_2.0",self.clik_file)
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False
        try:
            if self.lensing:
                self.clik = clik.clik_lensing(clik_file)
                try: 
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(clik_file)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            log.error(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated."
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.", self.name)
            raise HandledException
        except KeyError:
            log.error(
                "In the %s.data file, the field 'clik' of the "
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file", self.name)
            raise HandledException
        self.theory.needs({"l_max": self.l_max,
                           "Cl": ["TT", "TE", "EE", "BB"]})
        self.expected_params = list(self.clik.extra_parameter_names)
        # line added to deal with a bug in planck likelihood release:
        # A_planck called A_Planck in plik_lite
        if "plikHM_lite" in self.name:
            i = self.expected_params.index('A_Planck')
            self.expected_params[i] = 'A_planck'
            print("In %s, corrected nuisance parameter name A_Planck to A_planck" % self.name)
        
    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.theory.get_cl()
        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.params()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.params()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.expected_params))
        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb
                    index += self.clik.get_lmax()[i]+1
        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1
            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1
        # fill with likelihood parameters
        for i,p in enumerate(self.expected_params):
            tot[index+i] = params_values[p]
        # In case there are derived parameters in the future:
        # derived = params_values.get("derived")
        # if derived != None:
        #     derived["whatever"] = [...]
        # Compute the likelihood
        return self.clik(tot)[0]

    def close(self):
        del(self.clik) # MANDATORY: forces deallocation of the Cython class
        # Actually, it does not work for low-l likelihoods, which is quite dangerous!
