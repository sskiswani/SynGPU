Neural Networks on the GPU
===

A port of the SynfireGrowth Neural Network trial runner to the GPU, using CUDA.

Choose your flavor:

- CPU build and run: `cd ./port && make synfire && ./synfire`
- GPU build and run: `cd ./port && make cu_synfire && ./cu_synfire`

Use `make <rule> CXX=g++44` if you happen to be working on `*.cs.fsu.edu`.

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
- [ ] SynapticPlasticity (SP) kernel.
- [ ] Spike Loop (SL) kernel.
- [ ] Timestep (TS) kernel.
- [ ] Add atomic operation safety.
- [ ] Remove all traces of `./port/src/gpu/cudaTester.cu`
- [ ] Finish the report.
- [ ] Fix `run.py` (and/or make it useful)
- [ ] Submit

#Deliverables

- Tar of `port/*`
- PDFs of `rpt/proposal.tex` and `rpt/report.tex`.
- This README file.
- `run.py`

#Misc

Code exclusive to the GPU lives in the `port/src/gpu` folder.

~~`run.py` can be used to streamline usage, with the default execution pushing, compiling, and executing the source code. Although, it may not work universally due library dependencies.
```$ python3 run.py [-h] username password host [via-host]```~~ 

Arguments saved in a text file can be referenced by prefixing the filename with an `@` symbol, e.g. `python3 run.py @args.txt`.

##Options and Flags
The following options were ported from the original source.

- `-A <int>` (*default: 200000*) sets the numbers of trials before termination of the run 
- `-B <int>` (*default: 10*) sets the number of training neurons 
- `-C <float>` (*default: 2000*) sets the number of ms per trial 
- `-c <double>` (*default: .999996*) sets the decay rate of the synapses 
- `-D <int>` (*default: 10*) sets the max number of supersynapses a neuron may have
- `-d` (*default: false*) disable output
- `-f <double>` (*default: .1*) sets the fraction of synapses initially active 
- `-g <float>` Sets GLTP
- `-i <double>` (*default: .3*) sets the amplitude of the global inhibition 
- `-j <float>` sets the inhibition delay
- `-m <double>` (*default: 40Hz*) sets the excitatory spontaneous frequency 
- `-n <double>` (*default: 200Hz*) sets the inhibitory spontaneous frequency 
- `-o <double>` (*default: 1.3*) sets the amplitude of the spontaneous excitation 
- `-p <double>` (*default: .1*) sets the amplitude of the spontaneous inhibition 
- `-q <double>` (*default: .2*) sets the active syn threshold 
- `-r <double>` (*default: .4*) sets the super syn threshold 
- `-s <double>` (*default: .6*) sets the synapse maximum 
- `-u <double>` set maximum inhibitory synaptic strength
- `-x <double>` (*default: 1500 Hz*) sets spike rate of training excitation , input assumed to be in milliseconds.
- `-y <double>` (*default: .7*) sets amplitude of training excitation 
- `-z <double>` (*default: 8 ms*) sets the training excitation duration 

#References

Jun, Joseph K. AND Jin, Dezhe Z. [Development of Neural Circuitry for Precise Temporal Sequences through Spontaneous Activity, Axon Remodeling, and Synaptic Plasticity](http://dx.plos.org/10.1371/journal.pone.0000723) (2007)

```
@article{10.1371/journal.pone.0000723,
    author = {Jun, Joseph K. AND Jin, Dezhe Z.},
    journal = {PLoS ONE},
    publisher = {Public Library of Science},
    title = {Development of Neural Circuitry for Precise Temporal Sequences through Spontaneous Activity, Axon Remodeling, and Synaptic Plasticity},
    year = {2007},
    month = {08},
    volume = {2},
    url = {http://dx.plos.org/10.1371/journal.pone.0000723},
    pages = {e723},
    abstract = {<p>Temporally precise sequences of neuronal spikes that span hundreds of milliseconds are observed in many brain areas, including songbird premotor nucleus, cat visual cortex, and primary motor cortex. Synfire chains—networks in which groups of neurons are connected via excitatory synapses into a unidirectional chain—are thought to underlie the generation of such sequences. It is unknown, however, how synfire chains can form in local neural circuits, especially for long chains. Here, we show through computer simulation that long synfire chains can develop through spike-time dependent synaptic plasticity and axon remodeling—the pruning of prolific weak connections that follows the emergence of a finite number of strong connections. The formation process begins with a random network. A subset of neurons, called training neurons, intermittently receive superthreshold external input. Gradually, a synfire chain emerges through a recruiting process, in which neurons within the network connect to the tail of the chain started by the training neurons. The model is robust to varying parameters, as well as natural events like neuronal turnover and massive lesions. Our model suggests that long synfire chain can form during the development through self-organization, and axon remodeling, ubiquitous in developing neural circuits, is essential in the process.</p>},
    number = {8},
    doi = {10.1371/journal.pone.0000723}
}
```