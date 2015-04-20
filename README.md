Neural Networks on the GPU
===

A port of the SynfireGrowth neural network to the GPU, using CUDA.

`run.py` can be used to streamline usage, with the default execution pushing, compiling, and executing the source code.
```$ python27 run.py [-h] username password host [via-host]```

Arguments saved in a text file can be referenced by prefixing the filename with an `@` symbol, e.g. `python27 run.py @args.txt`.

#TODO

- [ ] Useful organization + separation of concerns
- [ ] Write the CUDA loaders
- [ ] Keep CMake and make files consistent with each other.
- [ ] Fill this out

#Deliverables

- Source Code in `port/` folder.
- PDFs of the project proposal and report.
- This README file.
- `run.py`

#Misc

Code exclusive to the GPU lives in the `port/src/gpu` folder.