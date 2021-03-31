import os, sys, glob, time
from optparse import OptionParser

import numpy as np
import torch
import time
import os

if __name__=="__main__":
    assert torch.cuda.is_available()
    
    ## This is where the input data comes from
    all_labels_path = ""
    all_spectra_path = ""
    ## This is where outputs will go
    workpath = ""
    
    neuron_type = "relu"
    num_neurons = 300
    learning_rate = 1e-4
    num_steps = 2e4
    batch_size = 512
    training_batch_size = 2048
    validation_batch_size = 512
    traintest_seed = 98457
    lnames = ["Teff","logg","vt","MH","aFe","CFe"]
    

    ## Import the code for the relevant network architecture
    if neuron_type == "relu":
        from The_Payne import training
    elif neuron_type == "sigmoid":
        from The_Payne import training_sigmoid as training
    elif neuron_type == "sigmoid2":
        from The_Payne import training_sigmoid2 as training
    print("Using neuron type",neuron)
    
    ## Set working directory
    os.chdir(workpath)
    print("Outputs will be written to",workpath)
    
    ## Setup train and test set
    all_labels = np.load(all_labels_path)
    N = all_labels.shape[0]
    assert all_labels.shape[1] == len(lnames), all_labels.shape[1]
    
    print("Loading all spectra (takes a minute)")
    start = time.time()
    all_spectra = np.load(all_spectra_path)
    assert all_spectra.shape[0] == N
    Npix = all_spectra.shape[1]
    print("Took {}s".format(time.time()-start))
    
    print("Seed:",traintest_seed)
    np.random.seed(traintest_seed)
    ii_train = np.ones(N, dtype=bool)
    ii_test = np.zeros(N, dtype=bool)
    ix_test = np.random.choice(np.arange(N), size=N//3, replace=False)
    ii_train[ix_test] = False
    ii_test[ix_test] = True
    
    training_labels = all_labels[ii_train,:]
    training_spectra = all_spectra[ii_train,:]
    test_labels = all_labels[ii_test,:]
    test_spectra = all_spectra[ii_test,:]
    print("Training labels", training_labels.shape)
    print("Training spectra", training_spectra.shape)
    print("Test labels", test_labels.shape)
    print("Test spectra", test_spectra.shape)
    
    ## Begin training!
    print("Begin training Payne")
    start = time.time()
    training.neural_net_loadbatch(training_labels, training_spectra,
                                  test_labels, test_spectra,
                                  num_neurons=num_neurons, learning_rate=learning_rate,
                                  num_steps=num_steps, batch_size=batch_size, num_pixel=Npix,
                                  training_batch_size=training_batch_size,
                                  validation_batch_size=validation_batch_size)
    print(f"Took {time.time()-start:.1f}s")
