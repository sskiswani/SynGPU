Neural Networks on the GPU
===

A port of the SynfireGrowth neural network to the GPU, using CUDA. Choose your flavor:

- CPU build and run: `cd ./port && make synfire && ./synfire`
- GPU build and run: `cd ./port && make cu_synfire && ./cu_synfire`

~~`run.py` can be used to streamline usage, with the default execution pushing, compiling, and executing the source code. Although, it may not work universally due library dependencies.
```$ python3 run.py [-h] username password host [via-host]```~~ 

Arguments saved in a text file can be referenced by prefixing the filename with an `@` symbol, e.g. `python3 run.py @args.txt`.

#TODO

- [x] Useful organization + separation of concerns.
- [x] Get SynfireGrowth working on the CPU locally.
- [x] Get SynfireGrowth working on `linprog.cs.fsu.edu`.
- [x] Get CUSynfire working on `gpu.cs.fsu.edu`.
- [x] Write the CUDA loaders.
- [x] Keep CMake and make files consistent with each other.
- [x] Fill this out.
- [x] SynapticDecay (SD) kernel.
- [ ] Membrane Potential Layer (MPL) kernel.
- [ ] Spike Loop (SL) kernel.
- [ ] Timestep (TS) kernel.
- [ ] Fix `run.py` (and/or make it useful)
- [ ] Finish the report.

#Deliverables

- Tar of `port/*`
- PDFs of `rpt/proposal.tex` and `rpt/report.tex`.
- This README file.
- `run.py`

#Misc

Code exclusive to the GPU lives in the `port/src/gpu` folder.