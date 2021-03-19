# payne_workflow
Workflow scripts for running and fitting the Payne for high-resolution spectra.

## Expected Steps/Modules
1. Synthesize Synthetic Grid. We will start with MOOG/ATLAS based on SMHR and eventually move to Turbospectrum/MARCS.
2. Train the Payne. Uses the library `The_Payne`
  2b. Check the Payne interpolation results, develop model masks
3. Use the Payne to fit high-res spectra. Uses the library `Payne4MIKE`
  3b. Check the Payne fits against standard stars and develop model masks.
4. Batch fit large numbers of spectra.
