# payne_workflow
Workflow scripts for running and fitting the Payne for high-resolution spectra.

## Expected Steps/Modules
0. Synthesize Synthetic Grid. We will start with MOOG/ATLAS based on SMHR and eventually move to Turbospectrum/MARCS.
   (This will happen eventually and probably will be mostly from a separate library.)

1. Train the Payne. Uses the library `The_Payne`
  1a. Train the Payne
  1b. Check the Payne interpolation results at a spectrum level
  1c. Check the Payne interpolation results by fitting actual labels
  1d. If the spectrum-level interpolation is good enough, add masks

2. Use the Payne to fit high-res spectra. Uses the library `Payne4MIKE`
  2a. Check the Payne fits against standard stars and develop model masks.

3. Batch fit large numbers of spectra.

