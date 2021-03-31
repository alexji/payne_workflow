import yaml

"""
Config keys:

label_names
all_labels_path
all_spectra_path
NN_path

payne_training:
  workpath
  neuron_type
  num_neurons
  learning_rate
  num_steps
  batch_size
  training_batch_size
  validation_batch_size
  seed

payne_eval_grid:
  workpath
  Nsave
  seed
  wranges
  tol

payne_gridfit:
  labelpath
  spectrapath
  outpath_labels
  outpath_spectra
  initial_stellar_labels

"""

def load_cfg(fname):
    with open(fname,"r") as fp:
        cfg = yaml.load(fp)
    print("Loaded cfg",fname)
    for section in cfg:
        print(section)
    return cfg

def load_cfg_key(fname, key):
    ## Need to figure out nested keys
    raise NotImplementedError
