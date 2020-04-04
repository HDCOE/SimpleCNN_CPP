#pragma pack(push,1)
/* Find the maximum value of tensor*/

float max(tensor_t<float>& in);
float average(tensor_t<float>& in);
tensor_t<float> create_mask(tensor_t<float> in, float value);
tensor_t<float> create_distribute(tensor_t<float> in, float value);
struct pool_layer
{
	tensor_t<float> input;
	tensor_t<float> output;
	tensor_t<float> gradient_dA;
	int stride;
	int poolsize;
	int mode;
	
	pool_layer(int _poolsize, int _stride, int _mode,  point_t in_size):
	// define input
	input(in_size.x, in_size.y, in_size.z),
	// define output
	output((in_size.x - _poolsize )/_stride + 1, (in_size.y - _poolsize )/_stride + 1 , in_size.z),	
	// gradient dA
	gradient_dA(in_size.x, in_size.y, in_size.z)
	{
		poolsize = _poolsize;
		stride = _stride;
		mode = _mode;
	}


void forward_pooling ()
{
	int n_w = output.size.x;
	int n_h = output.size.y;
	int n_c = output.size.z;

	int vert_start,vert_end,horiz_start,horiz_end;

	tensor_t<float> a_prev_temp(poolsize, poolsize, 1);

	for (int i = 0; i < n_h; ++i)
	{
		for (int j = 0; j < n_w; ++j)
		{
			for (int ch = 0; ch < n_c; ++ch)
			{
				    vert_start = i*stride;
                    vert_end = vert_start + poolsize;
                    horiz_start = j*stride;
                    horiz_end = horiz_start + poolsize;

                    a_prev_temp = copy_tensor(input, ch, vert_start, vert_end, horiz_start, horiz_end);

                    if (mode == 0) // max mode
                    {
                    	output(i,j,ch) = max(a_prev_temp);
                    }
                    else // average mode
                    {
                    	output(i,j,ch) = average(a_prev_temp);
                    }
			}
		}
	}
}

void Backward_pooling(tensor_t<float> dZ)
{
	int n_w = dZ.size.x;
	int n_h = dZ.size.y;
	int n_c = dZ.size.z;

	int vert_start,vert_end,horiz_start,horiz_end;

	tensor_t<float> a_prev_temp(poolsize, poolsize, 1);
	tensor_t<float> mask(poolsize, poolsize, 1);

	for (int i = 0; i < n_h; ++i)
	{
		for (int j = 0; j < n_w; ++j)
		{
			for (int ch = 0; ch < n_c; ++ch)
			{
				    vert_start = i*stride;
                    vert_end = vert_start + poolsize;
                    horiz_start = j*stride;
                    horiz_end = horiz_start + poolsize;

                    a_prev_temp = copy_tensor(input, ch, vert_start, vert_end, horiz_start, horiz_end);
                                     
                    if (mode == 0) // max mode, mask the maximum Ex: [1 2, 3 2] and dA[i,j] = 0.5 then mask = [0 0, 0.5 2]
                    {
                    	mask = create_mask(a_prev_temp,dZ(i,j,ch));              	
                    }
                    else // average mode, distribute Ex: dA[i,j] = 2 then dA_prev = [2/4 2/4, 2/4 2/4]
                    {
                    	mask = create_distribute(a_prev_temp,dZ(i,j,ch) / (poolsize * poolsize));
                    }
                    gradient_dA.save(mask, ch, vert_start, vert_end, horiz_start, horiz_end );
			}
		}
	}
}
};


float max(tensor_t<float>& in)
{
	float out = 0.0;
	for (int i = 0; i < in.size.y; ++i)
	{
		for (int j = 0; j < in.size.x ; ++j)
		{
			if (in(i,j,0) > out)
			{
				out = in(i,j,0);
			}
		}
	}
return out;

}
/* find the average of tensor */
float average(tensor_t<float>& in)
{
	float out = 0.0;
	for (int i = 0; i < in.size.y; ++i)
	{
		for (int j = 0; j < in.size.x ; ++j)
		{
			out += in(i,j,0);
		}
	}
return out/(in.size.y * in.size.x);
}

/* find the maximum element from input tensor and muliply with value
	[1 2; 2 4] and value =5 then create [0 0;0 1]*5 
 */
tensor_t<float> create_mask(tensor_t<float> in, float value)
{
	int w = in.size.x;
	int h = in.size.y;
	float max_val;
	max_val = max(in);

	tensor_t<float> out(w,h,1);


	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			if( in(i,j,0) == max_val)
				out(i,j,0) = value;
			else
				out(i,j,0) = 0;
		}
	}
	return out;
}

tensor_t<float> create_distribute(tensor_t<float> in, float value)
{
	int w = in.size.x;
	int h = in.size.y;
	float max_val;

	tensor_t<float> out(w,h,1);

	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			out(i,j,0) = value;
		}
	}
	//printf("This is create mask:");print_tensor(out);
	return out;
}

#pragma pack(pop)