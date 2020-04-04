#include <cassert>
#include <vector>

#include <cstdint>
#include <iostream>
#include <fstream>

#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include "byteswap.h"

#include "Tensor.h"
tensor_t<float> create(int w, int h, int ch)
{
	tensor_t<float> out (w,h,ch);
	return out;
}
int main()
{
	tensor_t<float> input (2,2,5);
	std::vector<tensor_t<float>> y_hat_vec ;
	
	y_hat_vec.insert(y_hat_vec.begin(), input);

	return 0;
}