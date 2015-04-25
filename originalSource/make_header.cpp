#include <stdlib.h>
//#include <string>
//#include <iostream>
#include <fstream>
using namespace std;


int main(int argc, char *argv[]){

  string fname = "synfireGrowth.h";
  ofstream writeheader;
  writeheader.open(fname.c_str());

  writeheader<<"#define SIZE "<<atoi(argv[1])<<endl;

  writeheader.close();
  return 0;
}
