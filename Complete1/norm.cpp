// normal_distribution
#include <iostream>
#include <string>
#include <random>

int main()
{
  const int nrolls=10;  // number of experiments
  const int nstars=10;    // maximum number of stars to distribute

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,1);


  int p[10]={};
  const double a = 12345;
 std::cout << generator << std::endl;

  for (int i=0; i<nrolls; ++i) {
    float number = distribution(generator);
    //if ((number>=0.0)&&(number<10.0)) ++p[int(number)];
 std::cout << number << std::endl;
   
  }

  for (int i=0; i<10; ++i) {
    std::cout << i << "-" << (i+1) << ": ";
    std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
  }

  return 0;
}