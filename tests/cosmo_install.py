if __name__ == "__main__":
    from cobaya.log import logger_setup
    logger_setup()
    from cobaya.conventions import _likelihood, _theory
    import sys
    info_install = {
        _theory: {"camb": None, "classy": None},
        _likelihood: {
            "planck_2015_lowl": None,
            "planck_2015_plikHM_TT": None,
            "planck_2015_lowTEB": None,
            "planck_2015_plikHM_TTTEEE": None,
            "planck_2015_lensing": None,
            "planck_2015_lensing_cmblikes": None,
            "bicep_keck_2015": None}}
    # Ignore Planck clik in Python 3
    if sys.version_info.major >= 3:
        for lik in info_install:
            if lik.startswith("planck_2015") and not lik.endswith("cmblikes"):
                info_install.pop(lik)
    path = sys.argv[1]
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    from cobaya.install import install
    install(info_install, path=path, no_progress_bars=True)
