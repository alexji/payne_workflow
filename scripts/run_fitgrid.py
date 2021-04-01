import time, os, glob, sys
import numpy as np

from payne_workflow.utils import load_nn
from payne_workflow.parse_config import *

from Payne4MIKE import utils
from Payne4MIKE import fitting_rpa1 as fitting

if __name__=="__main__":
    cfg = load_cfg(sys.argv[1])
    if len(sys.argv) == 5:
        ichunk = int(sys.argv[2])
        Nchunks = int(sys.argv[3])
        N_per_chunk = int(sys.argv[4])
        assert ichunk < Nchunks, (ichunk, Nchunks, N_per_chunk)
    else:
        ichunk = 0
        Nchunks = 1
        N_per_chunk = None
    
    all_labels_path = cfg["all_labels_path"]
    all_spectra_path = cfg["all_spectra_path"]
    NN_path = cfg["NN_path"]
    initial_stellar_labels = [float(x) for x in cfg["payne_fitgrid"]["initial_stellar_labels"]]
    
    workpath = cfg["payne_fitgrid"]["workpath"]
    if not os.path.exists(workpath): os.makedirs(workpath)
    
    ## Get the output paths such that they are sorted alphabetically
    zeropad = len(str(Nchunks))
    ichunk_str = (f"{{:0{zeropad}}}").format(ichunk)
    outpath_labels = os.path.join(workpath, f"bestfit_labels_{Nchunks}-{N_per_chunk}_{ichunk_str}.npy")
    outpath_spectra = os.path.join(workpath, f"bestfit_spectra_{Nchunks}-{N_per_chunk}_{ichunk_str}.npy")
    
    ## Load input data
    all_labels = np.load(all_labels_path)
    if N_per_chunk is None: N_per_chunk = Nspec
    all_spectra = np.load(all_spectra_path, mmap_mode="r")
    # Get indices of the data to read out
    i1 = N_per_chunk*ichunk
    i2 = N_per_chunk*(ichunk+1)
    Nallspec, Nlabels = all_labels.shape
    if i1 > Nallspec: raise ValueError(f"i1={i1} > Nallspec={Nallspec}")
    all_labels = all_labels[i1:i2]
    all_spectra = all_spectra[i1:i2]
    Nspec, Nlabels = all_labels.shape
    assert all_spectra.shape[0] == Nspec
    
    ## Read NN
    NN_coeffs, wavelength_payne = load_nn(NN_path)
    wavelength = wavelength_payne.reshape((1,-1))
    errors_payne = np.zeros_like(wavelength_payne)
    coeff_poly = 1
    num_order, num_pixel = 1, len(wavelength_payne)
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.
    
    ## Initialize p0
    assert len(initial_stellar_labels) == Nlabels, (Nlabels, initial_stellar_labels)
    print("Initializing stellar parameters at",initial_stellar_labels)
    normalized_stellar_labels = utils.normalize_stellar_parameter_labels_rpa1(
        initial_stellar_labels, NN_coeffs)
    p0_initial = np.zeros(6 + coeff_poly*num_order + 1 + 1)
    p0_initial[:6] = normalized_stellar_labels
    p0_initial[6::coeff_poly] = 1
    p0_initial[7::coeff_poly] = 0
    p0_initial[8::coeff_poly] = 0
    p0_initial[-2] = 0.5
    p0_initial[-1] = 0.0
    
    ## Do a quick time test
    Ntimetest = 10
    start = time.time()
    for i in range(Ntimetest):
        labels = all_labels[0].copy()
        norm_labels = utils.normalize_stellar_parameter_labels_rpa1(labels, NN_coeffs)
        full_labels = np.array(list(norm_labels) + [1, 1, 0])
        out = fitting.evaluate_model(full_labels, NN_coeffs, wavelength_payne, errors_payne, coeff_poly,
                                     wavelength, num_order, num_pixel, wavelength_normalized)[0]        
    dt = time.time()-start
    print(f"Quick check: time per iteration {dt*1000/Ntimetest:.0f}ms")
    sys.stdout.flush()
    
    ## Loop over and fit
    # This "2" is something that needs to be changed later
    # But it has to go into Payne4MIKE first, so whatevs for now
    # TODO There's a lot of hardcoded stuff here which is fine for now
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
