import time, os, glob, sys
import numpy as np

from .utils import load_nn

from Payne4MIKE import utils
from Payne4MIKE import fitting_rpa1 as fitting

if __name__=="__main__":
    nn_path = ""
    labelpath = ""
    spectrapath = ""
    assert os.path.exists(labelpath), labelpath
    assert os.path.exists(spectrapath), spectrapath

    outpath_labels = ""
    outpath_spectra = ""
    print("Writing to",outpath_labels)
    print("Writing to",outpath_bestfit)
    
    initial_stellar_labels = [5000, 2.0, 2.0, -2.5, 0.2, 0.0]
    
    ## Load input data
    all_labels = np.load(labelpath)
    Nspec, Nlabels = all_labels.shape
    all_spectra = np.load(spectrapath)
    assert all_spectra.shape[0] == Nspec
    
    ## Read NN
    NN_coeffs, wavelength_payne = load_nn(nn_path)
    wavelength = wavelength_payne.reshape((1,-1))
    errors_payne = np.zeros_like(wavelength_payne)
    coeff_poly = 1
    num_order, num_pixel = 1, len(wavelength_payne)
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.
    
    ## Initialize p0
    assert len(initial_stellar_labels) == Nlabels, Nlabels
    print("Initializing stellar parameters at",initial_stellar_labels)
    normalized_stellar_labels = utils.normalize_stellar_parameter_labels_rpa1(
        initial_stellar_labels, NN_coeffs)
    p0_initial = np.zeros(6 + coeff_poly*num_order + 1 + 1)
    p0_initial[:6] = normalized_stellar_parameters
    p0_initial[6::coeff_poly] = 1
    p0_initial[7::coeff_poly] = 0
    p0_initial[8::coeff_poly] = 0
    p0_initial[-2] = 0.5
    p0_initial[-1] = 0.0
    
    ## Do a quick time test
    Ntimetest = 10
    start = time.time()
    for i in range(Ntimetest):
        labels = all_labels[0]
        norm_labels = utils.normalize_stellar_parameter_labels_rpa1(labels, NN_coeffs)
        full_labels = np.array(list(norm_labels) + [1, 1, 0])
        out = fitting.evaluate_model(full_labels, NN_coeffs, wavelength_payne, errors_payne, coeff_poly,
                                     wavelength, num_order, num_pixel, wavelength_normalized)[0]        
    dt = time.time()-start
    print(f"Quick check: time per iteration {dt*1000/Ntimetest:.0f}ms")
    sys.stdout.flush()

    ## Loop over and fit
    bestfit_labels = np.zeros((Nspec, Nlabels+2))
    bestfit_spectra = np.zeros((Nspec, num_pixel))
    
    for i in range(Nspec):
        start = time.time()
        spectrum = all_spectra[i:i+1]
        spectrum_err = np.zeros_like(spectrum) + 0.001
        spectrum_blaze = np.ones_like(spectrum)
        
        RV_array = np.array([0.0])
        bounds = np.zeros((2, 8))
        bounds[0,:6] = -0.5
        bounds[1,:6] =  0.5
        bounds[0,-2] = 0.01
        bounds[1,-2] = 1.
        bounds[0,-1] = -10.
        bounds[1,-1] = 10.        
        iorder = 0
        
        popt_best, model_spec_best, chi_square = fitting.fitting_mike(
            spectrum, spectrum_err, spectrum_blaze, wavelength,
            NN_coeffs, wavelength_payne,
            errors_payne=errors_payne,
            p0_initial=p0_initial,
            bounds_set=bounds,
            RV_prefit=False, blaze_normalized=False,
            RV_array=RV_array,
            polynomial_order=coeff_poly-1,
            order_choice=[iorder])
        
        popt_print = utils.transform_coefficients_rpa1(popt_best, NN_coeffs)
        bestfit_labels[i,0:Nlabels] = popt_print[0:Nlabels]
        bestfit_labels[i,-2:] = popt_print[-2:]    
        
        bestfit_spectra[i] = model_spec_best

        dt = time.time()-start
        print(f"--------- Fitting {i} took {dt:.1f}s")
        fmt1 = "{:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0"
        fmt2 = "{:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} {:.1f}"
        print(f"  Input:", fmt1.format(*all_labels[i]))
        print(f"  Output:", fmt2.format(*bestfit_labels[i]))
        sys.stdout.flush()
    np.save(outpath_labels, bestfit_labels)
    np.save(outpath_spectra, bestfit_spectra)
