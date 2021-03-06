from __future__ import division
import numpy as np
import os
from copy import deepcopy

from cobaya.conventions import _theory, _likelihood, _params, _derived_pre, _prior
from cobaya.conventions import _sampler, _chi2, separator, _path_install, _p_ref
from cobaya.run import run
from cobaya.yaml import yaml_load
from cobaya.input import get_default_info, merge_params_info


# Tolerance for the tests of the derived parameters, in units of the sigma of Planck 2015
tolerance_derived = 0.055


# Converting 100cosmomc_theta to cosmomc_theta in Planck's covmats #######################

def adapt_covmat(filename, tmpdir, theory="camb", theta_factor=100):
    with open(filename, "r") as original:
        params = original.readline()[1:].split()
        covmat = np.loadtxt(filename)
    i_logA = params.index("logA")
    params[i_logA] = "logAs1e10"
    i_theta = params.index("cosmomc_theta")
    if theory == "camb":
        params[i_theta] = "cosmomc_theta_100"
    elif theory == "classy":
        params[i_theta] = "theta_s_100"
    # if used for cosmomc_theta or theta_s, not their multiples
    if theta_factor != 100:
        covmat[i_theta, :] /= (100/theta_factor)
        covmat[:, i_theta] /= (100/theta_factor)
    filename_new = os.path.join(str(tmpdir),"covmat.dat")
    np.savetxt(filename_new, covmat, fmt="%.8g", header=" ".join(params))
    return filename_new


# Body of the best-fit test ##############################################################

def body_of_test(modules, best_fit, info_likelihood, info_theory, ref_chi2,
                 best_fit_derived=None):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: info_theory,
            _likelihood: info_likelihood,
            _sampler: {"evaluate": None}}
    # Add best fit
    info[_params] = merge_params_info(*(
        [yaml_load(baseline_cosmology)] +
        [get_default_info(lik, _likelihood).get(_params, {})
         for lik in info[_likelihood]]))
    for p in best_fit:
        try:
            if _prior in info[_params].get(p, {}):
                info[_params][p]["ref"] = best_fit[p]
        except:  # was fixed or derived
            info[_params][p] = best_fit[p]
    # We'll pop some derived parameters, so copy
    derived = deepcopy(baseline_cosmology_derived)
    best_fit_derived = deepcopy(best_fit_derived)
    if list(info[_theory].keys())[0] == "classy":
        # More stuff that CLASS needs for the Planck model
        info[_params].update(baseline_cosmology_classy_extra)
        # Remove "cosmomc_theta" in favour of "H0" (remove it from derived then!)
        info[_params].pop("cosmomc_theta")
        info[_params].pop("theta")
        info[_params]["H0"] = {_prior: {"min": 0, "max": 100}, _p_ref: best_fit["H0"]}
        derived.pop("H0")
        if best_fit_derived is not None:
            best_fit_derived.pop("H0", None)
            # Don't test those that have not been implemented yet
            for p in ["zstar", "rstar", "thetastar", "DAstar", "zdrag",
                      "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq",
                      # BBN!!!
                      "DH", "Y_p"]:
                derived.pop(p)
                best_fit_derived.pop(p, None)
    else:
        info[_theory]["camb"] = info[_theory]["camb"] or {}
        info[_theory]["camb"].update({"stop_at_error": True})
    # Add derived
    if best_fit_derived is not None:
        info[_params].update(derived)
    updated_info, products = run(info)
    # Check value of likelihoods
    for lik in info[_likelihood]:
        chi2 = products["sample"][_chi2+separator+lik][0]
        assert abs(chi2-ref_chi2[lik]) < ref_chi2["tolerance"], (
            "Testing likelihood '%s': | %g - %g | = %g >= %g"%(
                lik, chi2, ref_chi2[lik], abs(chi2-ref_chi2[lik]), ref_chi2["tolerance"]))
    # Check value of derived parameters
    not_tested = []
    not_passed = []
    for p in best_fit_derived or {}:
        if best_fit_derived[p][0] is None or p not in best_fit_derived:
            not_tested += [p]
            continue
        rel = (abs(products["sample"][_derived_pre+p][0]-best_fit_derived[p][0]) /
               best_fit_derived[p][1])
        if rel > tolerance_derived*(
                2 if p in ("YHe", "Y_p", "DH", "sigma8", "s8omegamp5") else 1):
            not_passed += [(p, rel)]
    print("Derived parameters not tested because not implemented: %r" % not_tested)
    assert not not_passed, "Some derived parameters were off: %r" % not_passed


# Baseline priors ########################################################################

baseline_cosmology = r"""
omegabh2:
  prior:
    min: 0.005
    max: 0.1
  ref:
    dist: norm
    loc: 0.0221
    scale: 0.0001
  proposal: 0.0001
  latex: \Omega_\mathrm{b} h^2
omegach2:
  prior:
    min: 0.001
    max: 0.99
  ref:
    dist: norm
    loc: 0.12
    scale: 0.001
  proposal: 0.0005
  latex: \Omega_\mathrm{c} h^2
# If using CLASS, rename to "100*theta_s"!!!
cosmomc_theta: "lambda theta: 1.e-2*theta"
theta:
  prior:
    min: 0.5
    max: 10
  ref:
    dist: norm
    loc: 1.0411
    scale: 0.0004
  proposal: 0.0002
  latex: 100\theta_\mathrm{MC}
  drop: True
tau:
  prior:
    min: 0.01
    max: 0.8
  ref:
    dist: norm
    loc: 0.09
    scale: 0.01
  proposal: 0.005
  latex: \tau_\mathrm{reio}
logA:
  prior:
    min: 2
    max: 4
  ref:
    dist: norm
    loc:   3.1
    scale: 0.001
  proposal: 0.001
  latex: \log(10^{10} A_s)
  drop: True
As: "lambda logA: 1e-10*np.exp(logA)"
ns:
  prior:
    min: 0.8
    max: 1.2
  ref:
    dist: norm
    loc: 0.96
    scale: 0.004
  proposal: 0.002
  latex: n_\mathrm{s}
"""

baseline_cosmology_classy_extra = {"N_ur": 2.0328, "N_ncdm": 1,
                                   "m_ncdm": 0.06, "T_ncdm": 0.71611}

# Derived parameters, described in
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
baseline_cosmology_derived = {
    "H0":          {"latex": r"H_0"},
    "omegal":      {"latex": r"\Omega_\Lambda"},
    "omegam":      {"latex": r"\Omega_m"},
    "omegamh2":    {"derived": "lambda omegam, H0: omegam*(H0/100)**2", "latex": r"\Omega_m h^2"},
    "omegamh3":    {"derived": "lambda omegam, H0: omegam*(H0/100)**3", "latex": r"\Omega_m h^3"},
    "sigma8":      {"latex": r"\sigma_8"},
    "s8h5":        {"derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)", "latex": r"\sigma_8/h^{0.5}"},
    "s8omegamp5":  {"derived": "lambda sigma8, omegam: sigma8*omegam**0.5", "latex": r"\sigma_8 \Omega_m^{0.5}"},
    "s8omegamp25": {"derived": "lambda sigma8, omegam: sigma8*omegam**0.25", "latex": r"\sigma_8 \Omega_m^{0.25}"},
    "zrei":        {"latex": r"z_\mathrm{re}"},
    "As1e9":       {"derived": "lambda As: 1e9*As", "latex": r"10^9 A_s"},
    "clamp":       {"derived": "lambda As, tau: 1e9*As*np.exp(-2*tau)", "latex": r"10^9 A_s e^{-2\tau}"},
    "YHe":         {"latex": r"Y_P"},
    "Y_p":         {"latex": r"Y_P^\mathrm{BBN}"},
    "DH":          {"latex": r"10^5D/H"},
    "age":         {"latex": r"{\rm{Age}}/\mathrm{Gyr}"},
    "zstar":       {"latex": r"z_*"},
    "rstar":       {"latex": r"r_*"},
    "thetastar":   {"latex": r"100\theta_*"},
    "DAstar":      {"latex": r"D_\mathrm{A}/\mathrm{Gpc}"},
    "zdrag":       {"latex": r"z_\mathrm{drag}"},
    "rdrag":       {"latex": r"r_\mathrm{drag}"},
    "kd":          {"latex": r"k_\mathrm{D}"},
    "thetad":      {"latex": r"100\theta_\mathrm{D}"},
    "zeq":         {"latex": r"z_\mathrm{eq}"},
    "keq":         {"latex": r"k_\mathrm{eq}"},
    "thetaeq":     {"latex": r"100\theta_\mathrm{eq}"},
    "thetarseq":   {"latex": r"100\theta_\mathrm{s,eq}"},
    }
