import time, os, glob, sys
import shutil
import numpy as np

from payne_workflow.utils import load_nn
from payne_workflow.parse_config import *

from Payne4MIKE import utils
from Payne4MIKE import fitting_rpa1 as fitting

if __name__=="__main__":
    ## These are things that need to be specified in a config file
    cfg = load_cfg(sys.argv[1])
    
    all_labels_path = cfg["all_labels_path"]
    all_spectra_path = cfg["all_spectra_path"]
    NN_path = cfg["NN_path"]
    
    workpath = cfg["payne_evalgrid"]["workpath"]
    if not os.path.exists(workpath): os.makedirs(workpath)
    Nsave = cfg["payne_evalgrid"]["Nsave"]
    seed = cfg["payne_evalgrid"]["seed"]
    
    wranges = cfg["payne_evalgrid"]["wranges"]
    tol = cfg["payne_evalgrid"]["tol"]
    
    ## These are automatically generated paths
    os.chdir(workpath)
    orig_path = os.path.join(workpath,"original_NN_grid.npy")
    eval_path = os.path.join(workpath,"evaluate_NN_grid.npy")
    small_orig_path = os.path.join(workpath,"small_original_NN_grid.npy")
    small_eval_path = os.path.join(workpath,"small_original_NN_grid.npy")
    small_labels_path = os.path.join(workpath,"small_all_labels.npy")
    tolpath = os.path.join(workpath,f"thresh{tol*1000:03.0f}/")
    
    
    ## Load in NN info and set up evaluation
    NN_coeffs, wavelength_payne = load_nn(NN_path)
    wavelength = wavelength_payne.reshape((1,-1))
    errors_payne = np.zeros_like(wavelength_payne)
    coeff_poly = 1
    num_order, num_pixel = 1, len(wavelength_payne)
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    ## Save the wavelength and label arrays
    shutil.copyfile(all_labels_path, os.path.join(workpath, "all_labels.npy"))
    all_labels = np.load("all_labels.npy")
    np.save(os.path.join(workpath,"wave.npy"), wavelength_payne)
    
    ##### Evaluate Grid
    
    print("Starting to evaluate grid")
    start = time.time()
    new_spectra = np.zeros((all_labels.shape[0], num_pixel), dtype=np.float16)
    for i in range(all_labels.shape[0]):
        labels = all_labels[i]
        norm_labels = utils.normalize_stellar_parameter_labels_rpa1(labels, NN_coeffs)
        full_labels = np.array(list(norm_labels) + [1, 1, 0])

        out = fitting.evaluate_model(full_labels, NN_coeffs, wavelength_payne, errors_payne, coeff_poly,
                                     wavelength, num_order, num_pixel, wavelength_normalized)[0]
        new_spectra[i] = out.astype(np.float16)
        if i % 50 == 0:
            sys.stdout.write("{} {:.1f}s\r".format(i, time.time()-start))
            sys.stdout.flush()
    dt = time.time()-start
    N = all_labels.shape[0]
    print(f"Took {dt:.1f}s to evaluate {N} spectra, {dt*1000/N:.1f}ms/spectrum")
    print()
    print("Saving...")
    np.save(outpath, new_spectra)
    
    
    ## Extract a random subset of spectra for plotting purposes
    if not os.path.exists(orig_path):
        all_spectra = np.load(all_spectra_path)
        orig = all_spectra.astype(np.float16)
        np.save(orig_path, orig)
    else:
        orig = np.load(orig_path)
    np.random.seed(seed)
    ix = np.sort(np.random.choice(orig.shape[0], size=Nsave, replace=False))
    np.save(small_orig_path, orig[ix])
    np.save(small_eval_path, eval[ix])
    np.save(small_labels_path, all_labels[ix])
    print(f"Done! Saved {Nsave} spectra")
    
    
    ## Compare grids
    start = time.time()
    
    print("Loading")
    orig = np.load(orig_path)
    print("Loaded grid 1", time.time()-start)
    eval = np.load(eval_path)
    print("Loaded grid 2", time.time()-start)
    diff = eval - orig
    
    # Which pixels are substantially deviating
    badpix = np.abs(diff) > tol
    print("Computed badpix", time.time()-start)
    np.save(tolpath+"NN_grid_badpix.npy", badpix)
    print("Saved badpix", time.time()-start)
        
    # Not continuum pixel
    notcont = orig < 1 - tol
    print("Computed notcont", time.time()-start)
    np.save(tolpath+"NN_grid_notcont.npy", notcont)
    print("Saved notcont", time.time()-start)
    
    ## Save some overall outputs
    np.save(tolpath+"NN_grid_totalbadpix.npy", badpix.sum(axis=1))
    np.save(tolpath+"NN_grid_totalnotcont.npy", notcont.sum(axis=1))
    np.save(tolpath+"NN_grid_totalnotcontbadpix.npy", (notcont & badpix).sum(axis=1))
    print("Saved sums", time.time()-start)
    ## Save some outputs by wavelength
    for w1, w2 in wranges:
        ii = (wave > w1) & (wave <= w2)
        suffix = f"{w1/10}-{w2/10}"
        np.save(tolpath+f"NN_grid_totalbadpix_{suffix}.npy",
                badpix[:,ii].sum(axis=1))
        np.save(tolpath+f"NN_grid_totalnotcont_{suffix}.npy",
                notcont[:,ii].sum(axis=1))
        np.save(tolpath+f"NN_grid_totalnotcontbadpix_{suffix}.npy",
                (notcont & badpix)[:,ii].sum(axis=1))
        print("Saved", w1, w2, time.time()-start)
