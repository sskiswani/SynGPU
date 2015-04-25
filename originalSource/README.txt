Synfire Growth README
Created: 5/31/08
Author: Aaron J. Miller

1. Installation

...is exceedingly easy.  Just unpack the file to any directory you
choose.  The package includes only 4 files and a directory,
'Examples':

i) synfireGrowth.cpp contains all the c++ code for the simulation.
The details of the simulation are detailed in the publication:

Jun JK, Jin DZ (2007) "Development of Neural Circuitry for Precise
Temporal Sequences through Spontaneous Activity, Axon Remodeling, and
Synaptic Plasticity." PLoS ONE 2(8):
e723.doi:10.1371/journal.pone.0000723.

The code is separated into 4 sections:
		
a)The top portion of the file defines global variables, which are set
to the defaults detailed in Jun, Jin (2007). Any of the global
variables can be reset from the command line using the command line
flags listed in Part 2.

b)The following section is the declaration of the "neuron" class.
Integrate-and-fire neurons are what we use in this simulation, and the
"neuron" class includes all variables and subroutines relevant to
implementing integrate-and-fire neurons. See Jun, Jin (2007) for
details.

c)The next section declares the "synapses" class. The "synapses" class
tracks the strengths of all synapses during the simulation and is
responsible for their time evolution.

d)The final section is the main routine. It reads all runtime option
flags from the command line, creates the neccessary "synapses" and
"neuron" objects, loops through the trials and the timesteps, and
writes data files to the appropriate data directories.

ii) syn_analysis.m is a MATLAB file that creates plots from the data
that synfire_Growth.cpp produces. See Part 3.

iii) make_header.cpp is a short code that writes a header file used by
synfireGrowth.cpp that specifies the size of the network being
simulated. It is not necessary for the user to compile and run this
code, but it is neccessary to leave this file in the directory from
which the user creates new simulations.

iv) create_run is a shell script that sets up a new run so that the
user doesn't need to make the data subdirectories or complile and run
make_header.cpp manually.  The script prompts the user for the name of
a new directory for the new simulation, then prompts the user for the
size of the network that will be simulated. It creates the new
directory, copies synfireGrowth.cpp and syn_analysis.m to the new
directory, creates the header file synfireGrowth.h, and makes all the
neccessary data directories.  Lastly, it compiles synfireGrowth.cpp,
creating an executable with the same name as the simulation directory
entered by the user, so all the user needs to do is execute with the
runtime options.

v) Examples contains a set of results obtained from the simulation.

--------------------------------------------------------------------------------

2. Starting a simulation
	
i. Creating a new directory for the run
	
It is recommended that every simulation begin with executing the
script create_run. The script automatically sets up all data
subdirectories needed by the program synfireGrowth.cpp, and it creates
the appropriate header file for the simulation, after prompting the
user for the name of the new simulation directory and the size of the
network.  The script compiles all the code for the simulation.  **Some
compilers will give a single warning during compiling due to a double
that is converted to an int. This is not a problem; the code is
functioning properly.** Further details are documented above in
section 1.iv.

ii. Execute synfireGrowth with runtime options (code includes
additional options describe in the comments)

After running create_run, the code is properly compiled and the
simulation is ready to begin running. The name of the executable that
create_run makes is the same as the name of the simulation
directory. Unless the user has chosen to simulate a network of exactly
1000 neurons, the network size to which the default simulation
parameters are tuned, it is recommended that the user use the
following runtime options to adjust the parameters.  In particular,
the appropriate spontaneous excitatory activity frequency and
amplitude is very intimately related to the size of the network, and
must be adjusted for each new size. As a rule of thumb for setting the
values, synfire chains grow optimally in this model when the network
spikes around 200 times per trial and the average membrane potential
in the network is near -75mV. Other parameters may also be adjusted
yielding new and widely varying results.

****Parameter flags (refer to Jun, Jin (2007) for further details)

'-q <double>' sets the active synapse threshold (default is
.2). Synapses with strength larger than this threshold have a
physiological effect on their postsynaptic targets.

'-r <double>' sets the super synapse threshold (default is .4). The
number of synapses with strength above this threshold is limited by
axon remodeling (see also '-D')

'-s <double>' sets the synapse maximum (default is .6). Synapses are
capped at this value.

'-f <double>' sets the fraction of synapses initially active (default
is .1). In particular, we implement a step distribution initally for
simplicity, which is a feature of the simulation that should be
improved, as the distribution is not stable. The maximal allowed
inital synapse is set (see the "synapses" constructor) as half the
difference of the super-synapse and active-synapse thresholds.

'-a'

'-b'

'-c <double>' sets the decay rate of the synapses (default is
.999996). Synapses are reduced by this factor after every trial in
order to offset the synaptic potentiation that occurs as the network
is spontaneously excited. It is how we implement a slow "memory leak"
in the network.

'-x <double>' sets spike rate of training excitation (default is 1500
Hz). Generated from a Poisson distribution

'-y <double>' sets amplitude of training excitation (default is
.7). All spikes are of this strength.

'-z <double>' sets the training excitation duration (default is 8
ms). Always at the beginning of each trial.

'-m <double>' sets the excitatory spontaneous frequency (default is
40Hz). Generated from a Poisson distribution.

'-n <double>' sets the inhibitory spontaneous frequency (default is
200Hz). Generated from a Poisson distribution.

'-o <double>' sets the max amplitude of the spontaneous excitation
(default is 1.3). Generated from a uniform distribution.

'-p <double>' sets the max amplitude of the spontaneous inhibition
(default is .1). Generated from a uniform distribution.

 '-i <double>' sets the amplitude of the global inhibition (default is
 .3). After each network spike, all neurons are inhibited, to reduce
 computational complexity.

'-A <int>' sets the numbers of trials before termination of the run
(default is 200000).

'-B <int>' sets the number of training neurons (default is 10). Always
chosen with neuron labels 0,1,2,3...

'-C <int>' sets the number of ms per trial (default is 2000).

'-D <int>' sets the max number of supersynapses a neuron may have
(default is 10).

'-L' static synapses

****Log File
At the beginning of each run, synfireGrowth.cpp generates a log file
in the simulation directory: log<runid>.txt. The program documents in
this file the values of all adjustable simulation parameters. It also
contains a copy of the commands issues to it, in case the run fails
and must be called again.

****Data Output/Input Flags

'-# <int>' sets the ID number for the run (default is 1). It is very
####!!important!!#### that if you are loading a synaptic configuration
from a previous run that you change the runid or the simulation will
overwrite all previous data!

'-l <string>' loads synapse data from path <string>. The data file
must contain a SIZExSIZE array. Presumably, the <string> will be a
file from the "syn" subdirectory, unless the user generated the array
artifically.

'-d' turns on the output of various statistics for diagnostic purposes
to the screen after each trial. This is particularly helpful when
trying to determine if spontaneous activity levels are appropriate for
the selected network size.  ***Format:<trial> <spikes this trial>
<avg. membrane potential> <runtime this trial> <# active connections>

'-t <int>' saves synapse data every <int> trials (default is 100). The
simulation saves to "syn/syn<trial>r<runid>.dat" the entire
synaptic strength matrix as a SIZExSIZE array.  These files can be
read with syn_analysis.m to make plots of synaptic strengths
vs. trial, so the default save interval is set small to obtain maximal
resolution. Similarly these are the files syn_analysis.m reads to make
graphs of the distribution of weights.

'-u <int>' saves the spike data of the network every <int> trials
(default is 1000). The simulation saves to
"roster/roster<trial>r<runid>.dat" all spikes during the
trial. This file can be read with syn_analysis.m to make roster plots
of the network spikes.  ***Format: <label> <group_rank> <spike time>
[--includes spikes not in the chain]

'-v <int1> <int2>' tracks membrane voltage of neuron <int2> and one of
its postsynaptic neurons every <int1> trials (defaults are
<int1>=1000, <int2=0>). The data is Written to
"volt/volt_track_<pre>-<post>_<trial>r<runid>.dat" This file can
be used to generate plots of the membrane voltage and conductivities
using MATLAB.  <int1> also sets how often the membrane potentials and
the conductance values are saved at the end of the trial for
statistical purposes.  This data is written to the file
"volt/volt_ensemble<trial>r<runid>.dat.  ***Format: <volt.>
<ex. cond.> <inh. cond.> <volt.> <ex. cond.> <inh. cond.> <time>,
where the first set of voltage and conductances is that of the
"pre" neuron, specified by argument <int2>, and the second set
is that of the "post" neuron, which is chosen during runtime to
be the first neuron contained in pre's list of active synapses
("actsyn[pre]") that is not a training neuron. The format of the
data in the second (ensemble) files is the same except each line
contains the values from one neuron.

'-w <int>' saves network topology information in two formats (readable
and data) every <int> trials (default is 1000). The readable file
saves to "read/net_struct<trial>r<runid>.txt." The data file
writes to "connect/connect<trial>r<runid>.dat". It is read by
syn_analysis.m to create network topology plots.  ***Format: <pre>
<pre_group> <post> <post_group> <post_sat?> <G[pre][post]>, where
(__)_group are the group rankings which roughly correspond to firing
order, and post_sat? is either 1 or 0 according to whether the post
neuron is saturated or not.


****User-Defined Flags -- At the very top of the main routine is a
    large "switch" statement that reads the command line inputs. The
    first comment in the switch statement is "Default
    simulations/User-Defined Flags." This section is intended for
    user-defined "default" parameter sets so that user needs not enter
    long lists of flags once the user finds a parameter set suitable
    for the user's purposes.  Two examples have already been included:
    '-2' which sets suitable conditions for a 200 neuron network, and
    '-4' which sets suitable conditions for a 400 neuron network.

--------------------------------------------------------------------------------

3. Using MATLAB and syn_analysis.m

Assuming the simulation was initiated using the script create_run
included in the package, producing plots from the simulation is
extremely easy. Running syn_analyis.m with MATLAB from the directory
created for the simulation by create_run produces the following menu
of analysis selections:

>*****Generate plots from data files created by synfireGrowth.cpp.*****
>1. Draw network topology
>2. Display distribution of synaptic weights
>3. Generate roster plot\n
>4. Plot synaptic weights over range of trials
>0. Quit
>Please select one of the above options or to quit:

Examples of the plots that each of these options produces are included
in the 'Examples' directory bundled in the package, but a short
description of each is included below.

i)Network topology -- These plots are the best visualization of
synfire chain that grows during the simulation. The lines in the plot
are supersynapses, the vertices are neurons whose labels are given by
the numbers shown. Green lines are forward connections from a neuron
in a group to a neuron in the next consecutive group, red lines are
forward connections to groups beyond the next consecutive group, and
blue lines are lateral or back connections (connections to neurons in
the same or earlier groups). Generally, the connections depicted are
the only supersynapses in the network, unless the spontaneous activity
levels are set too high or the synaptic decay factor is set too close
to 1. Shorter chains can spontaneously form if the spontaneous
activity is too high or the synapses aren't decaying quickly enough.

-Generating network plots is quite simple. syn_analysis.m will prompt
 the user for the trial and run ID of the data desired to generate the
 plot. It loads the plot data from the "connect" subdirectory of the
 simulation directory created by create_run. The program will
 determine the number of groups in the chain structure, and prompt the
 user for the range of groups to be displayed. After the plot is
 generated, the user has the option to generate other plots from the
 same data (presumably with a different range of groups).

-Generating sequences of network plots in order to visualize the
 formation of the chain is also possible with syn_analysis.m. Instead
 of entering a single trial at the prompt, the user can enter a MATLAB
 array of trials and the program will generate a network plot of each
 trial with a pause between each trial. The user advances the sequence
 by pressing any key.

Incidentally, the network of active connections can also be visualized
with the same software, although the process of displaying it is a bit
labor intensive, as the user must make a small adjustment to the code
before the run generating the data starts. In the piece of code that
writes the connection data (~line 1150), the "synapses" subroutines
"rank_groups" and "write_groups" take a char input 's'. Changing that
input to 'a' will inform the class to analyze the active connections
instead of the super connections.  This feature may have little
practical use, but it's worth noting.

ii)Synaptic weight distribution -- With these plots, the user can
visualize the distribution of all weights in the system. syn_analysis
prompts the user for the trial and run ID of the data to be visualized
and fetches the corresponding file from the "syn" subdirectory in the
simulation directory. It then prompts the user for the synaptic cap
for the run, which can be found in log<runid>.txt, and the size of the
"bins" for the distribution.

Warning: the sorting of the weights into the bins for visualization is
a computationally heavy process, especially if the network is large
(>800 neurons). It can take several minutes for MATLAB to generate the
plot, but one way to shorten the runtime is to increase the bin
size. Of course, increasing the bin size decreases the resolution;
however bin sizes up to .4 can still yield meaningful plots.

iii)Roster plot -- This plot displays all the spikes during a
trial. The user is prompted for the trial and run ID of the data to be
analyzed, and the program locates that data in the subdirectory
"roster" of the simulation directory. The user has two display
options: 1) Plot spikes by neuron label, 2) Plot spikes by group
number. Option 1) displays the spikes of each neuron on its own
horizontal line, whereas option 2) displays spikes from neurons in the
same group on the same horizontal line, and spikes from all neurons
which are not part of the chain on the time axis. Regardless of which
option is chosen, spikes from neurons which are not incorporated into
the chain are colored RED and spikes that are part of the chain are
colored BLUE.  The user can decide whether the whole trial should be
displayed, or only a range of times within the trial.

iv)Time-evolution of weights -- The plot displays the "random walk"
that the synaptic weights take during the simulation. The user is
first prompted for a MATLAB array of trials to display and the run
ID. Then the user is prompted for up to 100 pre/post neuron pairs
entered using square brackets: "[pre post]". syn_analysis then loads
each of the specified data files from the subdirectory "syn" in the
simulation directory and extracts the values G[pre][post] for each
specified pre/post pair.

Warning: this process can take several minutes, if the network size or
the range of trials is large.  If the range of trials to be displayed
is large (>10000), often sampling at intervales of 500 yields an
acceptable resolution for most purposes.

The different synapses are color-coded in the plot, but MATLAB by
default has only 9 different colors.
