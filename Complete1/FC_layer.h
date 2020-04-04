#include "Softmax.h"
#include <random>
/* we can do conv of x with  w which has same size as x,
 or we can stretch x to be vector and multiply with vector w */
#pragma pack(push,1)



struct  fc_layer
{
	tensor_t<float> input;
	std::vector<tensor_t<float>> W;
	tensor_t<float> bias;
	tensor_t<float> output;
	cache gradient;
	tensor_t<float> dZ;

	fc_layer(int out_size, point_t input_size):
	// initialize input
	input(input_size.x, input_size.y, input_size.z),
	// initialize output
	output(1,1,out_size),
	// initialize bias
	bias(1,1,out_size),
	// create gradient dA, dW, db
	gradient(input.size, input.size, out_size, bias.size),
	dZ(1,1,out_size)
	{
		//initialize Weight
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,1);
  float volume_size = input_size.x * input_size.y * input_size.z;
		
		// intialize bias
		bias = create_tensor(1,1,out_size,0);

		for (int i = 0; i < out_size; ++i)
		{
			tensor_t<float> W_t(input_size.x, input_size.y, input_size.z);
			//W_t = create_tensor(W_t.size.x, W_t.size.y, W_t.size.z, 0.5+ (float)i / 10);
			for (int c = 0; c < W_t.size.z; ++c)
			{
				for (int w = 0; w < W_t.size.x; ++w)
				{
					for (int h = 0; h <  W_t.size.y; ++h)
					{
						W_t(w,h,c) = distribution(generator) / sqrt(volume_size);// ((double) rand() / (RAND_MAX)) + (float)i /10 ;// random number between 0-1 //0.5f + (float)c / 10;
					}
				}
			}
			W.insert(W.begin()+i,W_t);
			//W.push_back(W_t);
		}
	}

void forward_fc() 
{
	int w = input.size.y;
	int h = input.size.x;
	int c = input.size.z;

	int n_c = (int)W.size();
	float dot_out = 0.0;

	for (int filter = 0; filter < n_c ; ++filter)
	{
		// this is dot product W dot X 
		dot_out = input.dot(W[filter]);

		output(0,0,filter) = (dot_out) + bias(0,0,filter);
	}
}

	//dZ = dZ = dA*backrelu(a)
	//dW = (1/m)*dZ.dot(x.T)
	//db = (1/m)*sum(dZ)
	//dX = dot(W.T,dZ)
void backward_fc(tensor_t<float> dZ)
{
	float m = (float) dZ.size.z;

	int n_h = input.size.x ; int n_w = input.size.y; int n_c = input.size.z;


	//dZ = dA;

	for (int filter = 0; filter < (int)W.size(); ++filter)
	{
		for (int i = 0; i < n_h; ++i)
		{
			for (int j = 0; j < n_w; ++j)
			{
				for (int ch = 0; ch < n_c; ++ch)
				{
					// dW = 1/m *dZ.dot(a_prev)
					gradient.dW[filter](i,j,ch) = (1/m) * dZ(0,0,filter) * input(i,j,ch);		

					// dA = 1/m * WT.dot(dZ)
					gradient.dA(i,j,ch) += (1/m) * W[filter](i,j,ch) * dZ(0,0,filter);
				}
			}
		}
		//calculate db
		 gradient.db(0,0,filter) += (1/m) * dZ(0,0,filter);
	}
}
void fc_weight_update()
{
    float learning_rate = 0.1;
	int n_h = W[0].size.x;
	int n_w = W[0].size.y;
	int n_c = W[0].size.z;

	for (int filter = 0; filter < (int)W.size(); ++filter)
	{
		//tensor_t<float> temp(n_h,n_w,n_c);

		for (int i = 0; i < n_h; ++i)
		{
			for (int j = 0; j < n_w; ++j)
			{
				for (int ch = 0; ch < n_c; ++ch)
				{
					W[filter](i,j,ch) = W[filter](i,j,ch) - learning_rate * gradient.dW[filter](i,j,ch);
				
					//W[filter](i,j,ch) = temp(i,j,ch);

				// clear gradient dW,db
					gradient.dW[filter](i,j,ch) = 0;
					gradient.db(0,0,filter) = 0;
					gradient.dA(i,j,ch) = 0;
				}
			}
		}
	}
}
};
#pragma pack(pop)