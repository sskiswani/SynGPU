
//Author: Aaron Miller
//Last Modified: 3/02/2010

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#include "synfireGrowth.h"
#include "microtime.h"
#include "ran1.h"

//random number seed
long seed = -100000;

double DT=.1; //timestep (ms)
double INV_DT=10; //(1/DT)

int runid =1;  //identifies the particular run
bool LOAD = false; //1 means load data
bool ILOAD = false;
int trials = 200000, trial_t = 2000, *train_lab; //default # of trials, length of trial (ms)

//Save strings
string loadpath, iloadpath, synapse_path="syn/syn", roster_path="roster/roster", volt_e_path="volt/volt_ensemble";
string volt_t_path="volt/volt_track_",conn_path="connect/connect", r_conn_path = "read/net_struct";
string dat_suff = ".dat", txt_suff = ".txt", sim_base=".", sim_lab;
int synapse_save = 100, roster_save = 1000, volt_save = 1000, conn_save = 1000; //save data intervals

//Tracking variables during trial
ofstream track_volt; //stream that tracks volt and conds during trial
int volt_post, volt_pre=0; //contains the label of the neuron whose voltage is being tracked, it postsynaptic to volt_pre

double t = 0.0; //current time in ms
int *whospike, whocount = 0;//tracks labels of neurons that spike during current step, length of whospike[]
int group_rank[SIZE]; //contains group ranking of each neuron after chain network forms

//Spontaneous activity defaults
double exfreq=40, infreq=200, examp=1.3, inamp=.1, global_i = .3, inh_d=0, leak = -85.0;

//Synapses defaults
int NSS = SIZE, tempNSS=10; //max # of supersynapses allowed
double act = .2, sup =.4, cap = .6, frac = .1, isynmax = .3, eq_syn = .3;
double syndec = .99999;
int conn_type=1;
bool plasticity=true;
double window=200; //history time window size (ms)

//Training defaults
int ntrain = 10, ntrg = 1; //number of training neurons
bool man_tt=false; //manual training time bool (false if training occurs at t=0)
double *train_times;
double training_f = 1.5; //training spike frequency (ms)^(-1)
double training_amp = .7; //training spike strength
double training_t = 8.0; //training duration in ms

//Stats
bool stats_on=true;
int sc=0; //total spike counter
double av = 0;
time_t rock, roll;

//-----------------------------------------------------------------------------------------------------

//Neuron object
class neuron {

 private:
	 // Two factors that limit development of cycles
	 //recount - refactory period 25 ms
	 //latcount - LTD (longterm depression) 20 ms
  double nvolt, gexh, ginh, *spkhist;//membrane potential; excitatory, inhibitory conductances; spike times
  int latcount, refcount, spkcount, label;//latent, refractory period counters; spike counter; neuron label
  double spexfreq, spinfreq, LEAKREV;//spontaneous excitatory, inhibitory frequencies
  double spexamp, spinamp;//amplitudes of excitatory, inhibitory excitations
  double global_inh; //amplitude of global inhibition

  //Integrate & Fire model parameters
  const static double INDECAY = 3.0, EXDECAY = 5.0, MEMDECAY=20.0;//inhibitory, excitatory, membrane time constants in ms
  const static double SPKTHRES = -50.0, RESET = -80.0;//spike threshold, membrane reset potentials in mV
  const static double INHIBREV = -75.0;//inhibitory reversal potentials in mV
  const static double REFTIME = 25.0, LATTIME = 2.0;//refractory, latency time intervals

 public:
  neuron (int lab, double e, double i, double ea, double ia, double g);
  void set_sp_freq(double e, double i) {spinfreq = i*.001; spexfreq = e*.001;}; //convert from Hz to (ms)^(-1)
  void resetval();
  void resetspike() {if(spkcount!=0) {spkcount = 0; delete[] spkhist;}};
  int get_spkhist(double* & spikes) {if (spkcount!=0) spikes=spkhist; else spikes=NULL; return spkcount;};
  int get_label() {return label;}; //return label
  double get_pvt_var(char code); //return private variables
  bool updatevolt (); //4th order RK algorithm updates membrane potentials and conductances
  void neur_dyn(bool no_volt);
  void recspike(double t); //record spike times in spkhist arrays
  void excite_inhibit(double amp, char p);//increment conductances
};

//constuctor
neuron::neuron (int lab, double e, double i, double ea, double ia, double g){

  set_sp_freq(e, i);
  spexamp=ea;
  spinamp=ia;
  global_inh = g;
  label = lab;
  latcount = 0;
  refcount = 0;
  spkcount = 0;
  resetval();
  LEAKREV = leak;

}

void neuron::neur_dyn(bool no_volt){
	/*
	// New
	double c1[3], c2[3], c3[3];
	double temp0, temp1, temp2;

	temp2 = (DT/MEMDECAY)*((LEAKREV-nvolt)-gexh*nvolt+ginh*(INHIBREV-nvolt));
	temp2 = temp2 - (int)no_volt * temp2;
	temp1 =-DT*ginh/INDECAY;
	temp0 =-DT*gexh/EXDECAY;
	c1[0] = temp0; c1[1] = temp1; c1[2] = temp2;

	temp2 =  (DT/MEMDECAY)*(LEAKREV-(nvolt+(temp2/2.0))-(gexh+temp1/2.0)*(nvolt+(temp2/2.0))+(ginh+temp1/2.0)*(INHIBREV-(nvolt+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -DT*(ginh+temp1/2.0)/INDECAY;
	temp0 = -DT*(gexh+temp0/2.0)/EXDECAY;
	c2[0] = temp0; c2[1] = temp1; c2[2] = temp2;

	temp2 =  (DT/MEMDECAY)*(LEAKREV-(nvolt+(temp2/2.0))-(gexh+temp1/2.0)*(nvolt+(temp2/2.0))+(ginh+temp1/2.0)*(INHIBREV-(nvolt+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -DT*(ginh+temp1/2.0)/INDECAY;
	temp0 = -DT*(gexh+temp0/2.0)/EXDECAY;
	c3[0] = temp0; c3[1] = temp1; c3[2] = temp2;

	temp2 =  (DT/MEMDECAY)*(LEAKREV-(nvolt+(temp2/2.0))-(gexh+temp1/2.0)*(nvolt+(temp2/2.0))+(ginh+temp1/2.0)*(INHIBREV-(nvolt+(temp2/2.0))));
	temp2 = temp2 -(int)no_volt * temp2;
	temp1 = -DT*(ginh+temp1/2.0)/INDECAY;
	temp0 = -DT*(gexh+temp0/2.0)/EXDECAY;

	nvolt += (c1[2]+2*c2[2]+2*c3[2]+temp2)/6.0;
	gexh += (c1[0]+2*c2[0]+2*c3[0]+temp0)/6.0;
	ginh += (c1[1]+2*c2[1]+2*c3[1]+temp1)/6.0;
	*/


  //update membrane potential,conductances with 4th order RK
  double c1[3], c2[3], c3[3], c4[3];
  c1[0]=-DT*gexh/EXDECAY;
  c1[1]=-DT*ginh/INDECAY;
  if (no_volt==false) c1[2] = (DT/MEMDECAY)*((LEAKREV-nvolt)-gexh*nvolt+ginh*(INHIBREV-nvolt));
  else c1[2]=0;

  c2[0] = -DT*(gexh+c1[0]/2.0)/EXDECAY;
  c2[1] = -DT*(ginh+c1[1]/2.0)/INDECAY;
  if (no_volt==false) c2[2] = (DT/MEMDECAY)*(LEAKREV-(nvolt+(c1[2]/2.0))-(gexh+c1[0]/2.0)*(nvolt+(c1[2]/2.0))+(ginh+c1[1]/2.0)*(INHIBREV-(nvolt+(c1[2]/2.0))));
  else c2[2]=0;

  c3[0] = -DT*(gexh+c2[0]/2.0)/EXDECAY;
  c3[1] = -DT*(ginh+c2[1]/2.0)/INDECAY;
  if (no_volt==false) c3[2] = (DT/MEMDECAY)*(LEAKREV-(nvolt+(c2[2]/2.0))-(gexh+c2[0]/2.0)*(nvolt+(c2[2]/2.0))+(ginh+c2[1]/2.0)*(INHIBREV-(nvolt+(c2[2]/2.0))));
  else c3[2]=0;

  c4[0] = -DT*(gexh+c3[0])/EXDECAY;
  c4[1] = -DT*(ginh+c3[1])/INDECAY;
  if (no_volt==false) c4[2] = (DT/MEMDECAY)*(LEAKREV-(nvolt+c3[2])-(gexh+c3[0])*(nvolt+(c3[2]))+(ginh+c3[1])*(INHIBREV-(nvolt+c3[2])));
  else c4[2]=0;

  nvolt += (c1[2]+2*c2[2]+2*c3[2]+c4[2])/6.0;
  gexh += (c1[0]+2*c2[0]+2*c3[0]+c4[0])/6.0;
  ginh += (c1[1]+2*c2[1]+2*c3[1]+c4[1])/6.0;
}

bool neuron::updatevolt(){

  bool isspike = false;

  //spontaneous excitation and inhibition
  if (ran1(&seed)<DT*spexfreq) excite_inhibit(spexamp*ran1(&seed), 'e');
  if (ran1(&seed)<DT*spinfreq) excite_inhibit(spinamp*ran1(&seed), 'i');

  if (latcount<1 && refcount<1) {//if neuron isn't in latent period before spike

    neur_dyn(false);

    //neuron goes into latency before spike if neuron potential is greater than theshold & is not in refractory state
    if (nvolt>=SPKTHRES) {
      nvolt = 0; //spike
      latcount=(int)(LATTIME*INV_DT); //sets latency counter
    }

  }
  else{

    if(refcount>=1){
      refcount -= 1; //update refractory period counter
    }

    if(latcount>=1){//if neuron is in latency period before spike
	  latcount -= 1; //update latency counter
	  if(nvolt==0){
		nvolt = RESET;
	  }
	  if (latcount<1) {//if latent time period ends on this timestep, neuron spikes
		latcount=0; //reset counter (should already be zero, but just in case)
		refcount=(int)(REFTIME*INV_DT); //set refractory time counter
		isspike = true;
	  }
    }

    neur_dyn(true);

  }

  return isspike;
}

void neuron::excite_inhibit(double amp, char p){
  //p=='e' means excitatory, p=='i' means inhibitory
  switch (p) {
  case 'e': gexh+=amp;
    break;
  case 'i': ginh+=amp;
    break;
  default: cout<<"neuron.excite called with invalid char arg"<<endl; exit(1);
    break;
  }
}

double neuron::get_pvt_var(char code){
  //code = 'v' means membrane volt, code = 'e' means excitatory cond, code = 'i' means inhibitory cond.
  switch (code){
  case 'e': return gexh;
    break;
  case 'i': return ginh;
    break;
  case 'v': return nvolt;
    break;
  default: cout<<"neuron.get_pvt_var called with invalid char arg"<<endl; exit(1);
    break;
  }
}

void neuron::recspike(double t){

  if (spkcount!=0){
    double *temp = new double[spkcount]; //allocate temp array
    if (temp == NULL){
      cerr << "Memory allocation error in neuron.recspike" <<endl;
      exit(1);
    }
    //copy previous times to temp
    for (int i=0; i<spkcount; i++){
      temp[i]=spkhist[i];
    }

    delete[] spkhist; //deallocate old array
    spkhist = new double[spkcount + 1]; //allocate new (larger) array
    if (spkhist == NULL){
      cerr << "Memory allocation error in neuron.recspike" <<endl;
      exit(1);
    }
    //copy from temp to new array
    for (int i=0; i<spkcount; i++){
      spkhist[i] = temp[i];
    }
    delete [] temp; //deallocate temp array
  }
  else {
    spkhist = new double[1];
  }

  spkhist[spkcount] = t; //store new time
  spkcount++; //increment counter
}

void neuron::resetval(){ //randomly selects initial conditions before the beginning of each trial (see README)

  static double gexh_avg = .5*EXDECAY*spexamp*spexfreq;
  static double ginh_avg = .5*INDECAY*spinamp*spinfreq;

  nvolt = ran1(&seed)*(-55+80)-80;
  //select a conductance
  gexh =2*gexh_avg*ran1(&seed);
  ginh =2*ginh_avg*ran1(&seed);
}

//----------------------------------------------------------------------------------------------------------------

//Synapses Object
class synapses {

 private:

  double G[SIZE][SIZE]; //synaptic strength matrix
  int actcount[SIZE], supcount[SIZE], NSS[SIZE]; //arrays tracking numbers of active and super cns of each neuron in the network
  int *actsyn[SIZE], *supsyn[SIZE]; //arrays containing the postsynaptic cns of each neuron in the network
  double actthres, supthres, synmax; //active, super thresholds, synapse cap
  double syndec, GLTP; //synaptic decay
  //Synaptic plasticity parameters
  const static double ALTP = .01, ALTD = .0105; //ltp parameter, ltp amplitude, ltd parameter
  const static double INV_DECTLTP = .05, INV_DECTLTD = .05; //inverses of ltp, ltd decay times in inv(ms)
  const static double POTFUNT = 5.0, DEPFUNT = 5.25;

 public:
  synapses (double fract_act, double glob, double a, double s, double m, double d, int form_opt); //initialize new network;
  synapses (ifstream& synfile, double a, double s, double m, double d); //build synapses object from file
  void activate (char p, int pre, int post); //activate synapse G[pre][post]
  void deactivate (char p, int pre, int post); //deactivate synapse G[pre][post]
  double getsynstr(int pre, int post) {return G[pre][post];}; //return synaptic strength
  int getpost(char p, int pre, int* & post); //get postsynaptic neuron labels
  void synaptic_plasticity(int spiker, double t, neuron **net); //update synaptic strengths after network spikes
  void checkthres(double new_syn,int pre,int post,char p,char q); //check if synapse crosses threshold
  double potfun(double time, int spsc, double * hist, char p); //potentiation/depression function
  int count_syns(char p); //count the active or super synapses
  void synaptic_decay(); //scale down synapses by syndec
  void ordersyn(char p, int k); //order synapse list sequentially by label (for analysis purposes)
  void write_syns(ofstream& outfile); //write synaptic matrix to outfile
  void write_groups(ofstream& readfile, ofstream& datfile, int *group_rank, char p); //write connectivity to outfile

    void afferent(int * a, char p); //create afferent connection matrix
  int write_aff(ofstream& outfile, int * a, int post); //print afferent neuron labels of 'post' to outfile, given afferent connection matrix a

  // rankgroups and shortestpath not used
  void shortestpath(int * group, int group_size, int group_rank, int * res, char p); //compute shortest path to training neurons
  void rank_groups(int *group_one, int size_group_one, int *group_rank, char p); //determine group structure of network

  int findmode(int * vec, int ma); //determine group rank
  double getNSS(int lab) {return NSS[lab];};
};

synapses::synapses (double fract_act, double glob, double a, double s, double m, double d, int form_opt){

  actthres = a;
  supthres = s;
  synmax = m;
  syndec = d;
  GLTP = eq_syn;
  for(int i=0; i<SIZE; i++){
    NSS[i]=tempNSS;
  }

  switch(form_opt){

  case 1://randomly activate fract_act of all synapses

    for (int i=0; i<SIZE; i++){
      //cout<<i<<endl;
      actcount[i]=0;
      supcount[i]=0;
      for (int j=0; j<SIZE; j++){
	//cout<<"we are"<<endl;
        if (i!=j){
	  if (ran1(&seed)<=fract_act){
	    G[i][j]=actthres+(isynmax-actthres)*ran1(&seed);
	    activate('a',i,j);
	  }

	  else{
	    G[i][j]=actthres*ran1(&seed);
	  }
	  if(G[i][j]>synmax) G[i][j]=synmax;
	}
	else G[i][j]=0; //self synapses not allowed
      }

    }
    break;

  case 2:

    for (int i=0; i<SIZE; i++){
      actcount[i]=0;
      supcount[i]=0;
      for (int j=0; j<SIZE; j++){
	G[i][j]=glob;
      }
    }
    break;

  default:
    cout<<"Invalid form opt"<<endl;
    cout<<"add code to set an inital connectivity"<<endl;
    exit(0);
    break;
  }

  //int dum=0;
  //for (int i=0; i<SIZE; i++){
  //  dum += actcount[i];
  //}
  //cout<<"init act count = "<<dum<<endl;

}

synapses::synapses(ifstream& synfile, double a, double s, double m, double d) {

  actthres = a;
  supthres = s;
  synmax = m;
  syndec = d;
  for (int i=0; i<SIZE; i++){
    actcount[i]=0;
    supcount[i]=0;
    for (int j=0; j<SIZE; j++){
      synfile >> G[i][j];
      if (!synfile) {
	cerr << "load failed " <<i<<" "<<j<< endl;
	exit(1);
      }
      if (i==j) G[i][j]=0; //do not allow self connections
      if (G[i][j]>=actthres){
	activate('a',i,j);
	if (G[i][j]>=supthres){
	  activate('s',i,j);
	  if (G[i][j]>=synmax){
	    cerr<<"Saved matrix contains elements larger than synmax."<<endl;
	    exit(1);
	  }
	}
      }
    }
    if(supcount[i]>tempNSS){
      NSS[i]=supcount[i];
    }
    else{NSS[i]=tempNSS;}
  }
}

void synapses::activate (char p, int pre, int post) {
  //p == 'a' for active, p == 's' for super
  int *temp, *l, *m;
  switch (p){
  case 'a':
    l = &actcount[pre]; //l points to # of act synapses of pre
    if ((*l)!=0){//if old actsyn exists
      temp = new int[(*l)]; //make temp array
      if (temp == NULL){
	cerr << "Memory allocation error in neuron.activate" <<endl;
	exit(1);
      }
      for (int i=0;i<(*l);i++){
	temp[i]=actsyn[pre][i]; //store current values of actsyn[pre] in temp
      }
      delete[] actsyn[pre]; //deallocate actsyn[pre]
    }
    actsyn[pre] = new int[(*l)+1]; //allocate larger actsyn[pre]
    if (actsyn[pre] == NULL){
      cerr << "Memory allocation error in neuron.activate" <<endl;
      exit(1);
    }
    m = actsyn[pre]; //m points to new array
    break;

  case 's':
    l = &supcount[pre]; //l points to # of super synapses of pre
    if ((*l)!=0){//if old actsyn exists
      temp = new int[(*l)]; //make temp array
      if (temp == NULL){
	cerr << "Memory allocation error in neuron.activate" <<endl;
	exit(1);
      }
      for (int i=0;i<(*l);i++){
	temp[i]=supsyn[pre][i]; //store current values of supsyn[pre] in temp
      }
      delete[] supsyn[pre]; //deallocate supsyn
    }
    supsyn[pre] = new int[(*l)+1]; //allocate larger supsyn[pre]
    if (supsyn[pre] == NULL){
      cerr << "Memory allocation error in neuron.activate" <<endl;
      exit(1);
    }
    m = supsyn[pre]; //m points to new array
    break;

  default: cout<<"synapses.activate called with invalid char arg"<<endl;
    break;
  }

  if((*l)!=0){//if temp exists
    //copy old vals from temp into new array
    for (int i=0; i<(*l); i++){
      m[i]=temp[i];
    }
    delete [] temp;//deallocate temp
  }

  m[(*l)]=post;//place new synapse in new array
  (*l)++;//increment counter
}

void synapses::deactivate(char p, int pre, int post) {
  //p == 'a' active thres, p == 's'  for sup thres
  int *temp, *l, *m;
  switch (p){
  case 'a':
    l = &actcount[pre]; //l points to # of act synapses of pre
    if ((*l)!=1){//if the last post is not being deactivated
      temp = new int[(*l)]; //make temp array
      if (temp == NULL){
	cerr << "Memory allocation error in neuron.deactivate" <<endl;
	exit(1);
      }
      for (int i=0;i<(*l);i++){
	temp[i]=actsyn[pre][i]; //store current values of actsyn[pre] in temp
      }
    }
    delete[] actsyn[pre]; //deallocate actsyn[pre]
    if ((*l)!=1){//if the last post is not being deactivated
      actsyn[pre] = new int[(*l)-1]; //allocate smaller actsyn[pre]
      if (actsyn[pre] == NULL){
	cerr << "Memory allocation error in neuron.deactivate" <<endl;
	exit(1);
      }
      m = actsyn[pre]; //m points to new array
    }
    break;

  case 's':
    l = &supcount[pre]; //l points to # of super synapses of pre
    if ((*l)!=1){//if the last post is not being deactivated
      temp = new int[(*l)]; //make temp array
      if (temp == NULL){
	cerr << "Memory allocation error in neuron.deactivate" <<endl;
	exit(1);
      }
      for (int i=0;i<(*l);i++){
	temp[i]=supsyn[pre][i]; //store current values of supsyn[pre] in temp
      }
    }
    delete[] supsyn[pre]; //deallocate supsyn
    if ((*l)!=1){//if the last post is not being deactivated
      supsyn[pre] = new int[(*l)-1]; //allocate smaller supsyn[pre]
      if (supsyn[pre] == NULL){
	cerr << "Memory allocation error in neuron.deactivate" <<endl;
	exit(1);
      }
      m = supsyn[pre]; //m points to new array
    }
    break;

  default: cout<<"synapses.deactivate called with invalid char arg"<<endl;
    break;
  }

  if ((*l)!=1){//if the last post is not being deactivated
    //don't include post when placing values in temp back into supsyn[pre], so move the last value into its place
    for(int i=0; i<(*l); i++){
      if(temp[i]==post){
	temp[i]=temp[(*l)-1];
	break;
      }
    }
    (*l)--; //decrease counter

    //copy old vals from temp into new array
    for (int i=0; i<(*l); i++){
      m[i]=temp[i];
    }

    delete [] temp;//deallocate temp
  }
  else{//if the last post is being deactivated
    (*l)--;
  }
}

int synapses::getpost(char p, int pre, int* & post){

  int res = -1;
  switch(p){
  case 'a':
    post = actsyn[pre];
    res=actcount[pre];
    break;
  case 's':
    post = supsyn[pre];
    res=supcount[pre];
    break;
  default: cout<<"synapses.getpost called with invalid char arg"<<endl;
    exit(1);
  }
  return res;
}

void synapses::synaptic_plasticity(int spiker, double t, neuron **net){

  double * spk_times;
  int spk_count;
  double tempPot, tempDep, GPot, GDep;
  for (int k = 0; k<SIZE; k++){
    if(k!=spiker){

      GPot = G[k][spiker];
      GDep = G[spiker][k];
      spk_count = net[k]->get_spkhist(spk_times);
      if(spk_count!=0 && spk_times[spk_count-1]+window>=t){
	tempPot = GPot+ALTP*GLTP*potfun(t, spk_count, spk_times, 'p'); //potentiation
	tempDep = GDep*(1-ALTD*potfun(t, spk_count, spk_times, 'd')); //depression

	if(tempPot>synmax) tempPot=synmax;
	if(tempDep<0) tempDep=0;

	//Potentiate G[k][spiker]
	if(supcount[k]<NSS[k] || (supcount[k]==NSS[k] && GPot>=supthres)){
	  checkthres(tempPot,k,spiker,'a','p');
	  checkthres(tempPot,k,spiker,'s','p');
	  G[k][spiker] = tempPot;
	}
	//Depress G[spiker][k]
	if(supcount[spiker]<NSS[spiker] || (supcount[spiker]==NSS[spiker] && GDep>=supthres)){
	  checkthres(tempDep,spiker,k,'a','d');
	  checkthres(tempDep,spiker,k,'s','d');
	  G[spiker][k] = tempDep;
	}
      }
    }
  }
}

double synapses::potfun(double time, int spsc, double * hist, char p) {
  //p=='p' for potentiation, p=='d' for depression
   double res=0.0;
   double delt,pwt,inv_dect;
   double a=0.0;

   switch(p){
   case 'p':
     pwt=POTFUNT;
     inv_dect=INV_DECTLTP;
     break;
   case 'd':
     pwt = DEPFUNT;
     inv_dect=INV_DECTLTD;
     break;
   default: cout<<"Invalid input to synapses.potfun"<<endl;
     break;
   }

   for(int i=spsc-1; i>=0; i--){
     delt = time - hist[i];
     //if(p=='d'){
     //cout<<delt<<endl;
     //}
     if(delt<=window){
       if (delt <= pwt){
	 a = delt/pwt;
       }
       else {
	 a = exp(-(delt - pwt)*inv_dect);
       }
       res += a;
     }
     else break;
   }
  return res;
}

void synapses::checkthres(double new_syn, int pre, int post, char p, char q){
  //p=='a'/p=='s', check if G crossed active/super threshold
  //q == 'p' for potentiation, q == 'd' for depression
  double thres;
  switch(p){
  case 'a': thres = actthres;
    break;
  case 's': thres = supthres;
    break;
  default: cout<<"Invalid input for p in synapses.checkthres"<<endl;
    break;
  }

  switch(q){
  case 'p':
    if(G[pre][post]<thres && new_syn>=thres) activate(p, pre, post);
    break;
  case 'd':
    if(G[pre][post]>=thres && new_syn<thres) deactivate(p, pre, post);
    break;
  default: cout<<"Invalid input for q in synapses.checkthres"<<endl;
    break;
  }
}

void synapses::synaptic_decay(){
  for (int i=0; i<SIZE; i++){
    for (int j=0; j<SIZE; j++){
      checkthres(G[i][j]*syndec,i,j,'a','d');
      checkthres(G[i][j]*syndec,i,j,'s','d');
      G[i][j]*=syndec;
    }
  }

}

int synapses::count_syns(char p){

  int sum=0;
  switch(p){
  case 'a':
    for (int i=0; i<SIZE; i++){
      sum += actcount[i];
    }
    break;
  case 's':
    for (int i=0; i<SIZE; i++){
      sum += supcount[i];
    }
    break;
  default:cout<<"Invalid input to synapses.count_syns"<<endl;
    break;
  }

  return sum;
}

void synapses::write_syns(ofstream& outfile){

  for (int i=0; i<SIZE; i++){
    for (int j=0; j<SIZE; j++){
      outfile<<G[i][j]<<" ";
    }
    outfile<<endl;
  }
}

void synapses::write_groups(ofstream& readfile, ofstream& datfile, int* group_rank, char p){

  int c, *post, *throwaway, max_rank=0;
  for (int k=0; k<SIZE; k++){
    if (group_rank[k]>max_rank){
      max_rank = group_rank[k];//compute max_rank
    }
  }

  //Readable file
  switch (p){
  case 'a':
    readfile<<"##Active synapse network##"<<endl;
    break;
  case 's':
    readfile<<"##Super synapse network##"<<endl;
    break;
  default:cout<<"synapses.write_groups called with invalid char"<<endl;
    break;
  }
  for (int i=1; i<=max_rank; i++){
    readfile<<"GROUP "<<i<<":"<<endl;
    for (int j=0; j<SIZE; j++){
      if (group_rank[j]==i){
	readfile<<j<<": ";
	ordersyn('s',j);
	c = getpost(p,j,post);
	for (int k=0; k<c; k++){
	  readfile<<post[k]<<"(G"<<group_rank[post[k]]<<") ";
	}
	readfile<<endl;
      }
    }
    readfile<<endl;
  }

  //Data file
  for (int i=0; i<SIZE; i++){
    if (group_rank[i]!=0){
      c = getpost(p,i,post);
      for (int j=0; j<c; j++){
	datfile<<i<<" "<<group_rank[i]<<" "<<post[j]<<" "<<group_rank[post[j]]<<" "<<(getpost(p,post[j],throwaway)==tempNSS)<<" "<<G[i][post[j]]<<endl;
      }
    }
  }
}

  void synapses::rank_groups(int *group_one, int size_group_one, int *group_rank, char p){

  int aff[SIZE][SIZE], short_rank[SIZE];
  //Initialize these arrays
  for (int i=0; i<SIZE; i++){
    short_rank[i]=-1;
    group_rank[i]=-1;
    for (int j=0; j<SIZE; j++){
      aff[i][j]=0;
    }
  }

  afferent(&aff[0][0], p); //aff[c][d] contains 1 if G[d][c] is a supersynapse
  //short_rank[i] contains the length of the shortest path via synapses of type 'p' from neuron i to a neuron in group_one
  shortestpath(group_one,size_group_one,0,short_rank,p);

  for(int i=0; i<SIZE; i++){
    for (int j=0; j<SIZE; j++){
      aff[i][j]*=(short_rank[j]+1); //rank the afferent connections
    }
    //compute group_rank by finding mode of the short_rank of all afferent neurons
    group_rank[i] = findmode(aff[i],SIZE/size_group_one)+1;
  }

  for (int i=0; i<size_group_one; i++){
    group_rank[group_one[i]]=1; //set group_one's member's group_rank to one
  }
}

void synapses::afferent(int * a, char p){
  //p=='a' for active connections, p=='s' for super connections

  //aff[c][d] contains 1 if G[d][c] is a supersynapse
  switch (p){
  case 'a':
    for (int i=0; i<SIZE; i++){
      for (int j=0; j<actcount[i]; j++){
	*(a+SIZE*actsyn[i][j]+i)=1;
      }
    }
    break;
  case 's':
    for (int i=0; i<SIZE; i++){
      for (int j=0; j<supcount[i]; j++){
	*(a+SIZE*supsyn[i][j]+i)=1;
      }
    }
    break;
  default: cout<<"Invalid input to synapses.afferent."<<endl;
    break;
  }
}

int synapses::write_aff(ofstream& outfile, int * a, int post){
  int y=0; //tracks total # of afferent neurons
  for (int i=0; i<SIZE; i++){
    if (*(a+SIZE*post+i)!=0){
      y++;
      outfile<<i<<" ";
    }
  }
  outfile<<endl;
  return y;
}

void synapses::ordersyn(char p, int k){
  //p is 'a' for active, 's' for sup
  int count,temp;
  int * me;

  temp=0;

  switch(p){
  case 'a':
    count = actcount[k];
    me = actsyn[k];
    break;
  case 's':
    count = supcount[k];
    me = supsyn[k];
    break;
  default:cout<<"Invalid input to synapses.ordersyn"<<endl;
    break;
  }

  for (int i=0; i<(count-1); i++){
    for (int j=0; j<(count-1); j++){
      if(me[j]>me[j+1]){
	temp = me[j];
	me[j]=me[j+1];
	me[j+1]=temp;
      }
    }
  }
}

void synapses::shortestpath(int * group, int group_size, int group_rank, int * res, char p){
  //group is a set of <group_size> neurons which are a shortest distance of <group_rank> from group_one
  //p=='a' means follow all active connections, p=='s' means follow only superconnections
  for (int i=0; i<group_size; i++){
    if ((res[group[i]]>group_rank || res[group[i]]==-1)){//shortest_path passed through "res" must be initialized to -1
      res[group[i]]=group_rank;//change current value of shortest_path if current path is shorter

      switch(p){
      case 's':
	if (supcount[group[i]]!=0){
	  shortestpath(supsyn[group[i]],supcount[group[i]],group_rank+1,res,p);
	}
	break;

      case 'a':
	if (actcount[group[i]]!=0){
	  shortestpath(actsyn[group[i]],actcount[group[i]],group_rank+1,res,p);
	}
	break;

      default: cout<<"Invalid char input to synapses.shortestpath."<<endl;
	break;
      }
    }
  }
}

int synapses::findmode(int * vec, int ma){
//determines group rank of a neuron whose afferent connections are contains in vec (length SIZE)
  int res[ma];
  for (int i=0; i<ma; i++){
    res[i]=0;
  }
  int mode = -1;
  int count = -1;
  for (int i=0; i<SIZE; i++){
    if(vec[i]!=0){
      res[vec[i]-1]++;
      if (res[vec[i]-1]>count){
	count = res[vec[i]-1];
	mode = vec[i];
      }
    }
  }

  //check for tie, goes to lowest group number
  int check=0;
  for (int i=0; i<ma; i++){
    check += (int)(res[i]==count);
  }

  if (check>=2){
    for (int i=0; i<ma; i++){
      if (res[i]==count){
	mode = i+1;
	break;
      }
    }
  }

  return mode;
}

//-------------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {

  //set arguments passed in command line to override defaults
  for (int i=1; i<argc; i++){

    string commnd = argv[i];
    if (commnd[0] != '-') continue;
    else{
      switch (commnd[1]){

	//"DEFAULT" SIMUATIONS/USER-DEFINED FLAGS (in addition to the absolute default:1000 neuron network)
      case '2': //'-2' sets the default parameters for a typical 200 neuron network
	exfreq = 30, examp=1.88, syndec = .99992;
	break;
      case '4':
	exfreq = 33, examp=1.57, syndec = .99994;
	break;
	//OUTPUT & ANALYSIS FLAGS
      case 'd': // '-d' outputs various statistics after each trial for diagnostic purposes (i.e. spk count, avg volt, etc)
	stats_on = 1;
	break;
      case 't': // '-t <int>' saves synapse data every <int> trials (default is 1000)
	synapse_save = atoi(argv[i+1]);
	roster_save = synapse_save;
	break;
      case 'v':// '-v <int1> <int2>' tracks membrane voltage of neuron <int2> and one of its postsynaptic neurons
	       // every <int1> trials (default is int1=1000, int2=0 (training neuron))
	volt_save = atoi(argv[i+1]);
	if (argv[i+2][0]!='-') volt_pre = atoi(argv[i+2]);
	break;
      case 'w':// '-w <int>' saves connectivity info in readable format and also in a format suitable for analysis (default is 1000)
	conn_save = atoi(argv[i+1]);
	break;
	//PARAMETER FLAGS
      case '#': // '-# <int>' sets the ID number for the run (default is 1)
	runid = atoi(argv[i+1]);
	break;
      case 'A':// '-A <int>' sets the numbers of trials before termination of the run (default is 200000)
	trials = atoi(argv[i+1]);
	break;
      case 'B':// '-B <int>' sets the number of training neurons (default is 10)
	ntrain = atoi(argv[i+1]);
	break;
      case 'C':// '-C <int>' sets the number of ms per trial (default is 2000);
	trial_t = atoi(argv[i+1]);
	break;
      case 'D':// '-D <int>' sets the max number of supersynapses a neuron may have
	tempNSS = atoi(argv[i+1]);
	break;
      case 'F':
	ntrg = atoi(argv[i+1]);
	train_times = new double[ntrg+1];
	man_tt=true;
	for(int j=0; j<ntrg; j++){
	  train_times[j]=atof(argv[i+j+2]);
	}
	train_times[ntrg]=trial_t+1000;
	break;
      case 'J'://load inhibitory weights from file
	ILOAD=true;
	iloadpath = argv[i+1];
	break;
      case 'L':// synapses are static
	plasticity=false;
	break;
      case 'R'://set leak reversal (default is -85)
	leak = atof(argv[i+1]);
	leak = leak*(-1);
	break;
      case 'X': //base directory
	sim_base = argv[i+1];
	break;
      case 'Y': //numeric directory label
	sim_lab = argv[i+1];
	break;
      case 'a':// sets connectivity type (see README)
	conn_type = atoi(argv[i+1]);
	break;
      case 'c':// '-c <double>' sets the decay rate of the synapses (default is .999996)
	syndec = atof(argv[i+1]);
	break;
      case 'f':// '-f <double>' sets the fraction of synapses initially active (default is .1)
	frac = atof(argv[i+1]);
	if (frac>=1 || frac<0){
	  cerr << "Command line input -f is invalid, must be <1 and >=0" << endl;
	  exit(1);
	}
	break;
      case 'g': //sets GLTP
	eq_syn = atof(argv[i+1]);
	break;
      case 'i':// '-i <double>' sets the amplitude of the global inhibition (default is .3)
	global_i = atof(argv[i+1]);
	break;
      case 'j':// sets the inhibition delay
	inh_d = atof(argv[i+1]);
	break;
      case 'l': // '-l <string>' loads synapse data from path <string>
	LOAD = 1;
	loadpath = argv[i+1];
	break;
      case 'm': // '-m <double>' sets the excitatory spontaneous frequency (default is 40Hz)
	exfreq = atof(argv[i+1]);
	break;
      case 'n':// '-n <double>' sets the inhibitory spontaneous frequency (default is 200Hz)
	infreq = atof(argv[i+1]);
	break;
      case 'o':// '-o <double>' sets the amplitude of the spontaneous excitation (default is 1.3)
	examp = atof(argv[i+1]);
	break;
      case 'p':// '-p <double>' sets the amplitude of the spontaneous inhibition (default is .1)
	inamp = atof(argv[i+1]);
	break;
      case 'q':// '-q <double>' sets the active syn threshold (default is .2)
	act = atof(argv[i+1]);
	break;
      case 'r':// '-r <double>' sets the super syn threshold (default is .4)
	sup = atof(argv[i+1]);
	break;
      case 's':// '-s <double>' sets the synapse maximum (default is .6)
	cap = atof(argv[i+1]);
	break;
      case 'u'://set maximum inhibitory synaptic strength
	isynmax = atof(argv[i+1]);
	break;
      case 'x': // '-x <double>' sets spike rate of training excitation (default is 1500 Hz)
	training_f = atof(argv[i+1])*.001;//convert to (ms)^(-1)
	break;
      case 'y': // '-y <double>' sets amplitude of training excitation (default is .7)
	training_amp = atof(argv[i+1]);
	break;
      case 'z': // '-z <double>' sets the training excitation duration (default is 8 ms)
	training_t = atof(argv[i+1]);
	break;
      default:
	cout<<"Warning: command line flag "<< commnd <<" is not recognized."<<endl;
	break;

      }
    }
  }

  //Create neurons and synapses
  synapses *connectivity;
  synapses *inhibition_strength;
  neuron *network[SIZE];
  int group_s = (int) SIZE/ntrg;

  //Seed random number generator & check RAND_MAX
  seed = time(NULL)*(-1);
  //ran1(&seed);

  //initialize synapse object
  if (LOAD == false){
    //cout<<"loading"<<endl;
    connectivity = new synapses(frac,0,act,sup,cap,syndec,conn_type);
    if (connectivity == NULL){
      cerr << "Memory allocation error while creating synapses object" <<endl;
      exit(1);
    }
  }

  if (LOAD == true){
    ifstream infile;
    infile.open(loadpath.c_str());
    if (!infile){
      cerr<<"Failed to open "<<loadpath<<" containing synapse matrix"<<endl;
      exit(1);
    }
    connectivity = new synapses(infile,act,sup,cap,syndec);
    if (connectivity == NULL){
      cerr << "Memory allocation error while creating synapses object" <<endl;
      exit(1);
    }
  }

  //inhibition
  if(ILOAD==false){
    inhibition_strength = new synapses(1,global_i,1,1,1,1,2);
    if (inhibition_strength == NULL){
      cerr << "Memory allocation error while creating inhibition object" <<endl;
      exit(1);
    }
  }

  if(ILOAD==true){
    ifstream infile;
    infile.open(iloadpath.c_str());
    if (!infile){
      cerr<<"Failed to open "<<iloadpath<<" containing synapse matrix"<<endl;
      exit(1);
    }
    inhibition_strength = new synapses(infile,1000,1000,1000,1);
    if (inhibition_strength == NULL){
      cerr << "Memory allocation error while creating synapses object" <<endl;
      exit(1);
    }
  }

  //initialize neuron objects
  for (int i=0; i<SIZE; i++){
    network[i]= new neuron(i,exfreq,infreq,examp,inamp,global_i);
    if (network[i] == NULL){
      cerr << "Memory allocation error while creating neuron object" <<endl;
      exit(1);
    }
  }

  if(man_tt==false){
    train_times = new double[2];
    train_times[0]=0.0;
    train_times[1]=trial_t+1000;
  }

  //initialize array of training neuron labels
  train_lab = new int[ntrain*ntrg];
  if (train_lab == NULL){
    cerr<<"Could not allocate memory for train_lab in main."<<endl;
    exit(1);
  }
  int tc=0;
  for(int j=0; j<ntrg; j++){
    for (int i=0; i<ntrain; i++){
      train_lab[tc]=j*group_s+i;
      tc++;
    }
  }

  //set up inhibition delay
  int dsteps = 1 + (int) INV_DT*inh_d;
  int inh[dsteps];
  for(int i=0; i<dsteps; i++){
    inh[i] = 0;
  }

 //Write simulation parameters in logfile before simulation starts
  stringstream id_num;
  id_num << runid;
  string fname = sim_base+sim_lab+"/log"+id_num.str()+".txt";

  ofstream writelog;
  writelog.open(fname.c_str());

  writelog<<"Run started with command line:"<<endl;
  for (int i=0; i<argc; i++){
    writelog<<argv[i]<<" ";
  }
  writelog<<endl;
  writelog<<"SIZE = "<<SIZE<<" :: Max # of supersynapses = "<<tempNSS<<endl;
  writelog<<"Trials = "<<trials<<" :: Trial length  = "<<trial_t<<"ms"<<endl;
  writelog<<"Timestep = "<<DT<<"ms"<<endl;
  writelog<<endl<<"Neuron class:"<<endl;
  writelog<<"Sp. excitatory freq. = "<<exfreq<<"Hz :: Sp. inhibitory freq = "<<infreq<<"Hz"<<endl;
  writelog<<"Sp. excitatory amp. = "<<examp<<" :: Sp. inhibitory amp = "<<inamp<<endl;
  writelog<<"Ext current = "<<(leak+85)<<endl;
  if(ILOAD=true){
    writelog<<"Inh loaded from "<<iloadpath;
  }
  else{
    writelog<<"Global inh. amp. = "<<global_i;
  }
  writelog<<":: Inh delay = "<<inh_d<<endl;

  writelog<<endl<<"Synapses class:"<<endl;
  writelog<<"Act. thres. = "<<act<<" :: Sup. thres. = "<<sup<<" :: Max. syn. str. = "<<cap<<endl;
  writelog<<"Synapse decay factor = "<<syndec<<" :: Frac. initially active = "<<frac<<endl;
  writelog<<"GLTP = "<<eq_syn<<endl;
  writelog<<endl<<"Training:"<<endl;
  writelog<<"Training neurons = ";
  for(int i=0; i<ntrain; i++){
    writelog<<train_lab[i]<<" ";
  }
  writelog<<endl<<"Training time =  ";
  writelog<<train_times<<" ";
  writelog<<endl<<"Training duration = "<<training_t<<"ms"<<" :: Training freq = "<<training_f*1000<<"Hz :: Training amp = "<<training_amp<<endl;
  writelog<<endl<<"Data output and analysis:"<<endl;
  if (LOAD ==1){
    writelog<<"Synapse data loaded from "<<loadpath<<endl;
  }
  else writelog<<"Synapses initialized with option "<<conn_type<<endl;

  writelog<<"Synapse data saved every "<<synapse_save<<" trials"<<endl;
  writelog<<"Spike data saved every "<<roster_save<<" trials"<<endl;
  writelog<<"Neuron microvariables tracked and saved every "<<volt_save<<" trials"<<endl;
  writelog<<"Connection data saved every "<<conn_save<<" trials"<<endl;

  writelog.close();

  int trial_steps = (int) trial_t*INV_DT;

  double tTa[10], tTSa[trial_steps], tSDa[10];
  double tT[3], tTS[3], tMPL[3], tSL[3],tSD[3];
  tMPL[2]=0, tSL[2]=0, tTS[2]=0, tSD[2]=0;

  for (int a=0; a<=10/*trials*/; a++){//****Trial Loop****//
tT[0] = microtime();
    rock = time(NULL);//start timer
    int train_time_counter=0;
    int train_group_lab = 0;

    //Save membrane potential data for a pre TN and a post
    //(FORMAT: <pre.nvolt> <pre.gexh> <pre.ginh> <post.nvolt> <post.gexh> <post.ginh> <t>)
    if ((a%volt_save)==0){
      stringstream id_num, trial_num, id_pre, id_post;
      id_num << runid;
      trial_num << a;

      //select neuron to track (which isn't receiving training spikes)
      int *post;
      int c = connectivity->getpost('a',volt_pre,post);
      for (int i=0; i<c; i++){
	if (post[i]>=ntrain){
	  volt_post = post[i];
	  break;
	}
      }
      id_pre << volt_pre;
      id_post << volt_post;

      string fname = sim_base+sim_lab+"/"+volt_t_path + id_pre.str() + "-" + id_post.str() + "_" + trial_num.str() + "r" + id_num.str() + dat_suff;
      track_volt.open(fname.c_str());
    }


tTS[0] = microtime();


    //At t=0, on with the trial!
    for (int i=0; i<trial_steps; i++) {//****Timestep Loop****//
      //Excite training neurons

      if (t>=train_times[train_time_counter]){//****Training Loop****//
      	for(int j=train_group_lab*ntrain; j<ntrain*(train_group_lab+1); j++){
      	  if(ran1(&seed)<training_f*DT){
      	    network[train_lab[j]]->excite_inhibit(training_amp, 'e');
      	    //cout<<"excited "<<train_lab[j]<<" at "<<t<<endl;
      	  }
      	}

	if(t>=(train_times[train_time_counter]+training_t)){
	  //cout<<train_group_lab<<" "<<t<<endl;
	  train_time_counter++;
	  train_group_lab = 1;//rand()%ntrg;
	}
      }

      //Track voltages and conductances
      if ((a%volt_save)==0){
	track_volt<<network[volt_pre]->get_pvt_var('v')<<" "<<network[volt_pre]->get_pvt_var('e')<<" ";
	track_volt<<network[volt_pre]->get_pvt_var('i')<<" "<<network[volt_post]->get_pvt_var('v')<<" ";
	track_volt<<network[volt_post]->get_pvt_var('e')<<" "<<network[volt_post]->get_pvt_var('i')<<" "<<t<<endl;
      }

tMPL[0] = microtime();

      //Update membrane potentials first, keep track of who spikes
      for (int j=0; j<SIZE; j++) {//****Membrane Potential Loop****
	if (network[j]->updatevolt()){//if neuron j spikes this timestep, increase the length of whospike, store label
	  if (whocount!=0){
	    int *temp = new int[whocount];  //temp array while whospike is destroyed
	    if (temp==NULL){
	      cerr<<"Too many spikes this timestep, could not allocate memory for whospike"<<endl;
	      exit(1);
	    }

	    for (int k=0; k<whocount; k++){
	      temp[k]=whospike[k]; //put spikes in temp
	    }
	    delete[] whospike;
	    whocount++;
	    whospike = new int[whocount]; //create longer whospike
	    if (whospike==NULL){
	      cerr<<"Too many spikes this timestep, could not allocate memory for whospike"<<endl;
	      exit(1);
	    }

	    for (int k=0; k<(whocount-1); k++){
	      whospike[k]=temp[k]; //put spikes back into whospike
	    }
	    delete[] temp; //destroy temp
	  }
	  else{ //if this is the first spike this timestep
	    whocount++;
	    whospike = new int[whocount];
	    if (whospike==NULL){
	      cerr<<"Too many spikes this timestep, could not allocate memory for whospike"<<endl;
	      exit(1);
	    }
	  }
	  whospike[whocount-1]=j; //store spiking neuron in longer array
	}
      }

tMPL[1] = microtime();
tMPL[2] += (tMPL[1]-tMPL[0]);
//cout << "MPL current: " << tMPL[1]-tMPL[0] << "MPL Total: " << tMPL[2];

tSL[0] = microtime();


      for (int j=0; j<whocount; j++){//****Spike Loop****
		int *send_to;//pointer to array containing post neurons
		int send_count;//number of post neurons receiving spike

		//cout<<whospike[j]<<" spikes!!! at "<<t<<endl;
		network[whospike[j]]->recspike(t); //record time of spike
		sc++; //keep track of total number of spikes this trial

		//Send spikes
		if (connectivity->getpost('s',whospike[j],send_to)==connectivity->getNSS(whospike[j])){ //check to see if spiking neuron is saturated
		  for(int k = 0; k<connectivity->getNSS(whospike[j]); k++){
			network[send_to[k]]->excite_inhibit(connectivity->getsynstr(whospike[j],send_to[k]),'e'); //send spikes along super synapses
		  }
		}
		else {//spiking neuron isn't saturated, send spikes along active connections
		  send_count = connectivity->getpost('a',whospike[j],send_to);
		  for(int k = 0; k<send_count; k++){
			network[send_to[k]]->excite_inhibit(connectivity->getsynstr(whospike[j],send_to[k]),'e');
		  }
		}

		//Plasticity and remodeling
		if(plasticity==true){
		  connectivity->synaptic_plasticity(whospike[j],t,network);
		}

      }//****Spike Loop****

tSL[1] = microtime();
tSL[2] += (tSL[1]-tSL[0]);
//cout << "SL current: " << tSL[1]-tSL[0] << " SL total:" << tSL[2] << endl;

	//inhibition
    inh[dsteps-1]=whocount;

    for (int z=0; z<whocount; z++){

      for (int j=0; j<SIZE; j++){
	network[j]->excite_inhibit(inhibition_strength->getsynstr(whospike[z],j),'i');
      }
      //cout<<t<<" "<<inh[0]*global_i<<endl;

    }

    for(int i=0; i<dsteps-1; i++){
      inh[i]=inh[i+1];
    }

    if (whocount!=0) {
      delete [] whospike; //reset whocount and destroy whospike at end of timestep
      whocount = 0;
    }
    t += DT; //increment t
tTS[1] = microtime();
tTSa[i] = (tTS[1] - tTS[0]);
    }//****Timestep Loop****
tSL[2] = 0;
tMPL[2]=0;    t=0.0; //reset timer
    seed = time(NULL)*(-1); //reseed
    ran1(&seed);

    //stop tracking volt when trial is done
    if ((a%volt_save)==0){
      track_volt.close();
    }
tSD[0] = microtime();

    if (plasticity==true){
      connectivity->synaptic_decay(); //synapses decay after each trial
    }
tSD[1] = microtime();
tSDa[a] = (tSD[1]-tSD[0]);

    if (stats_on==1){//(FORMAT: <trial> <spike total> <av. volt> <runtime> <# of active connections>)
      for (int i=0; i<SIZE; i++){
	av += network[i]->get_pvt_var('v');
      }
      roll = time(NULL);
      cout << a <<" "<< sc <<" "<< (av/SIZE) <<" "<<(roll-rock)<<" "<<connectivity->count_syns('a')<<endl;

      sc = 0;
      av = 0;
    }

    //SAVE DATA FILES AND WEIGHTS
    //Connectivity  files (FORMAT: <pre> <pre_group#> <post> <post_group#> <sat(1)/unsat(0)> <G[pre][post]>)
    if((a%conn_save)==0){
      /*
      stringstream id_num,trial_num;
      id_num << runid;
      trial_num << a;
      string fname = r_conn_path + trial_num.str() + "r" + id_num.str() + txt_suff;
      string fname2 = conn_path + trial_num.str() + "r" + id_num.str() + dat_suff;
      ofstream write_conn_read;
      ofstream write_conn_dat;
      write_conn_read.open(fname.c_str());
      write_conn_dat.open(fname2.c_str());
      connectivity->rank_groups(train_lab,ntrain,group_rank,'s');//group structure
      connectivity->write_groups(write_conn_read,write_conn_dat,group_rank,'s');
      write_conn_read.close();
      write_conn_dat.close();
      */
    }


    //micro-variable (end of trial) data file (FORMAT: <n.label> <membrane volt> <exc.cond.> <inh.cond.>)
    if((a%volt_save)==0){
      stringstream id_num, trial_num;
      id_num << runid;
      trial_num << a;
      string fname = sim_base+sim_lab+"/"+volt_e_path + trial_num.str() + "r" + id_num.str() + dat_suff;

      ofstream write_volt;
      write_volt.open(fname.c_str());
      for (int i=0; i<SIZE; i++){
	write_volt<<i<<" "<<network[i]->get_pvt_var('v')<<" "<<network[i]->get_pvt_var('e')<<" ";
	write_volt<<network[i]->get_pvt_var('i')<<endl;
      }
      write_volt.close();
    }

    //Save syn weights
    if((a%synapse_save)==0){
      stringstream id_num, trial_num;
      id_num << runid;
      trial_num << a;
      string fname = sim_base+sim_lab+"/"+synapse_path + trial_num.str() + "r" + id_num.str() + dat_suff;

      ofstream write_syn;
      write_syn.open(fname.c_str());
      connectivity->write_syns(write_syn);
      write_syn.close();
    }

    //Save spike times for roster plot (FORMAT: <n.label> <group rank (0 if not part of chain)> <spike t>)
    if((a%roster_save)==0){
      stringstream id_num, trial_num;
      id_num << runid;
      trial_num << a;
      string fname = sim_base+sim_lab+"/"+roster_path + trial_num.str() + "r" + id_num.str() + dat_suff;

      //if((a%synapse_save)!=0){//group_rank is required to make a meaningful roster plot
      //	connectivity->rank_groups(train_lab,ntrain,group_rank,'s');
      //}
      ofstream write_roster;
      double *times;
      write_roster.open(fname.c_str());
      for (int i=0; i<SIZE; i++){
	for (int j=0; j<network[i]->get_spkhist(times); j++){
	  write_roster<<i<<" "<<times[j]<<endl;
	}
      }
      write_roster.close();
    }

    //Reset neuron values for next trial
    for (int i=0; i<SIZE; i++){
      network[i]->resetspike();
      network[i]->resetval();
    }

  tT[1] = microtime();
  tT[2] = (tT[1]-tT[0]);
  tTa[a] = tT[2];



  }//end of simulation

double avgT = 0, avgTS=0, avgSD=0;
  for (int i = 0; i < 10; ++i) {
    cout << "Trial: " << i  << " Time: " << tTa[i] << " TrialStep: " << tTSa[i] << " Synaptic_Decay: " << tSDa[i] << endl;
    avgT += tTa[i];
    avgTS += tTSa[i];
    avgSD += tSDa[i];
  }

  cout << "Avg Trial: " << avgT/10 << " Avg TrialStep: " << avgTS/10 << " Avg Decay: " << avgSD/10 << endl;



  return 0;
}//end of main

