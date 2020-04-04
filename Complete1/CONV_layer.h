using namespace std;
#include <random>
#pragma pack(push, 1)
struct conv_layer
{
	
	tensor_t<float> input;
	std::vector<tensor_t<float>> W;
	tensor_t<float> bias;
	tensor_t<float> output;
	hyperparam h_param;
	cache gradient;

	tensor_t<float> dZ; // input gradient


	conv_layer(int Wsize, int Nfilter, int stride, int pad, point_t in_size):
	// define input 
	input(in_size.x, in_size.y, in_size.z),
	// create output size by h = h - f+2pad/stride+1, w = w - f+2pad/stride+1
	output((in_size.x - Wsize + 2*pad)/stride+1, (in_size.y - Wsize + 2*pad)/stride+1, Nfilter),
	// bias with deep Nfilter
	bias(1,1,Nfilter),
	// create gradient dA, dW, db
	gradient(input.size, input.size, Nfilter, bias.size),

	dZ(output.size.x, output.size.y, output.size.z)
	{		
		//initialize Weight
		//tensor_t<float> W_t(Wsize,Wsize,in_size.z);
		
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,1);
		// intialize bias
		bias = create_tensor(1,1,Nfilter,0);
		// initialize pad,stride	
		h_param.pad = pad;
		h_param.stride = stride;	
		float volume_size = Wsize * Wsize * Wsize;
		for (int i = 0; i < Nfilter; ++i)
		{
			//W_t = create_tensor(W_t.size.x, W_t.size.y, W_t.size.z, 0.5f + (float)i / 10);
			//W.insert(W.begin()+i,W_t);

			tensor_t<float> W_t(Wsize, Wsize, in_size.z);
			//W_t = create_tensor(W_t.size.x, W_t.size.y, W_t.size.z, 0.5+ (float)i / 10);
			for (int c = 0; c < W_t.size.z; ++c)
			{
				for (int w = 0; w < W_t.size.x; ++w)
				{
					for (int h = 0; h <  W_t.size.y; ++h)
					{
						W_t(w,h,c) = distribution(generator) * sqrt(2 / volume_size);//((double) rand() / (RAND_MAX)); //(float)i /10 ;// random number between 0-1 //0.5f + (float)c / 10;
					}
				}
			}
			W.insert(W.begin()+i,W_t);
			
		}
	}


float singleconv(tensor_t<float>& a, tensor_t<float>& W)
{
	int w = a.size.x;
	int h = a.size.y;
	int c = a.size.z;

	tensor_t<float> z(w,h,c);
	float sum=0;
   
		for (int row = 0; row < h; ++row)
		{
			for (int col = 0; col < w; ++col)
			{
				for (int ch = 0; ch < c; ++ch)
   				{
				z(row,col,ch)=a(row,col,ch)*W(row,col,ch);
				sum = sum+z(row,col,ch);
				}
			}
   		}
   return sum;
}
void Conv_forward()
{
	int stride = h_param.stride;
	int pad = h_param.pad;

	int w = input.size.x;
	int h = input.size.y;
	int c = input.size.z;

	int f = W[0].size.x;
	int n_c = W[0].size.z;

	int n_w = output.size.x;
	int n_h = output.size.y;

	int vert_start,vert_end,horiz_start,horiz_end;

	tensor_t<float> a_slice_prev_temp(f,f,1); 
	tensor_t<float> W_temp(f,f,1);
	tensor_t<float> a_slice_pad(w + 2 * pad, h + 2  * pad, c); // padding output

	// Padding input with zero
	a_slice_pad = Padding(input, h_param.pad);

for (int filter_N = 0; filter_N < (int)W.size(); ++filter_N)
{
	for (int i = 0; i < n_h; ++i)
	{
		for (int j = 0; j < n_w; ++j)
		{
			for (int ch = 0; ch < n_c; ++ch)
			{
				//Find the corners of the current "slice"
                    vert_start = i * stride;
                    vert_end = vert_start + f;
                    horiz_start = j * stride;
                    horiz_end = horiz_start + f;
                 //pick sliding slice of each channel   
                    a_slice_prev_temp = copy_tensor(a_slice_pad, ch, vert_start, vert_end, horiz_start, horiz_end);
                 //pick weight of each channel
                    W_temp = copy_tensor(W[filter_N],ch,0,f,0,f);

                    output(i,j,filter_N) = singleconv(a_slice_prev_temp,W_temp);
			}
			output(i,j,filter_N) += bias(0,0,filter_N);
		}
	}

}

}

void Conv_backward()
{
	int w = input.size.x;
	int h = input.size.y;
	int c = input.size.z;

	int f = W[0].size.x;
	int w_c = c;

	int n_w = dZ.size.x;
	int n_h = dZ.size.y;
	int n_c = dZ.size.z;

	int pad = h_param.pad;
	int stride = h_param.stride;

	int vert_start,vert_end,horiz_start,horiz_end;

	tensor_t<float> dA(w,h,c);
	tensor_t<float> dA_temp(w,h,c);
	tensor_t<float> dA_padd(w + 2 * pad, h + 2 * pad, c);
	tensor_t<float> dA_padd_temp(w + 2 * pad, h + 2 * pad, c);

	tensor_t<float> W_temp(f,f,1);
	tensor_t<float> dW_temp(f,f,1);
	tensor_t<float> dW_temp2(f,f,1);


	tensor_t<float> a_slice_prev_temp(f,f,1);
	tensor_t<float> a_slice_pad(w + 2 * pad, h + 2 * pad, c);

	//db = create_tensor(1,1,n_c,0);

	dA.set_all(0); //dA = create_tensor(w,h,c,0); //create output and define it to be all 0

	dA_padd = Padding(dA,pad); // padding dA
	a_slice_pad = Padding(input, pad);

	for (int i = 0; i < n_h; ++i)
	{
		for (int j = 0; j < n_w; ++j)
		{
			for (int ch = 0; ch < w_c; ++ch)
			{
				/* select sliding area */
				    vert_start = i;
                    vert_end = vert_start + f;
                    horiz_start = j;
                    horiz_end = horiz_start + f;

                    for (int filter = 0; filter < (int)W.size(); ++filter)
						{
							/* Calculate dA by dA += sum(W*dZ[h,w])*/
                  			dA_padd_temp = copy_tensor(dA_padd, ch, vert_start, vert_end, horiz_start, horiz_end);
                    		W_temp = mul_scalar(W[filter], ch, dZ(i,j,filter));
                    		dA_padd_temp = tensor_add(W_temp, dA_padd_temp);
                    		dA_padd.save(dA_padd_temp,ch, vert_start, vert_end, horiz_start, horiz_end);

                    		/* calculate dW += sum (a_slice*dZ[h,w]) */
                    		a_slice_prev_temp = copy_tensor(a_slice_pad, ch, vert_start, vert_end, horiz_start, horiz_end);
                   		    dW_temp = mul_scalar(a_slice_prev_temp,0,dZ(i,j,filter));
                   		    dW_temp2 = copy_tensor(gradient.dW[filter],ch,0,f,0,f);
                   		    dW_temp = tensor_add(dW_temp,dW_temp2);

                   		    gradient.dW[filter].save(dW_temp,ch,0,f,0,f);
                    	}    

			}
			/* calculate db = sum(dZ)*/
			for (int filter = 0; filter < (int)W.size(); ++filter)
			{
				gradient.db(0,0,filter) += dZ(i,j,filter); 	     
			}
		}
	}
	
	for (int ch = 0; ch < c; ++ch)
	{
		//dA (dA_padd,ch,pad , dA_padd.size.x - pad, pad, dA_padd.size.y - pad); // copy from da_padd but cut the pad
		dA_temp = copy_tensor(dA_padd,ch,pad, dA_padd.size.x - pad, pad, dA_padd.size.y - pad);
		dA.save(dA_temp,ch,0, w, 0, h);
	}
	this->gradient.dA = dA;
}

void conv_weight_update()
{
    float learning_rate = 0.1;
	int n_h = W[0].size.x;
	int n_w = W[0].size.y;
	int n_c = W[0].size.z;

	for (int filter = 0; filter < (int)W.size(); ++filter)
	{
		tensor_t<float> temp(n_h,n_w,n_c);

		for (int i = 0; i < n_h; ++i)
		{
			for (int j = 0; j < n_w; ++j)
			{
				for (int ch = 0; ch < n_c; ++ch)
				{
					temp(i,j,ch) = W[filter](i,j,ch) - learning_rate * gradient.dW[filter](i,j,ch);
				//	printf("weight update  %f - %f \n",W[filter](i,j,ch),gradient.dW[filter](i,j,ch) );
					W[filter](i,j,ch) = temp(i,j,ch);
				// clear gradient dW,db
					gradient.dW[filter](i,j,ch) = 0;
					gradient.db(0,0,filter) = 0;
				}
			}
		}
	}
}
};
#pragma pack(pop)