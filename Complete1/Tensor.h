
struct point_t
{
	int x, y, z;
};
struct  hyperparam
{
	int stride;
	int pad;
};


template<typename T>

struct tensor_t
{
	T * data;

	point_t size;

/*Define size of tensor */
	tensor_t( int _x, int _y, int _z )
	{
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}
	
/* Define element of tensor by tensor(x,y,z)= value */
	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z )
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );
		return data[_z * (size.x * size.y) + _y * (size.x) +_x];
	}

	T& set_all( float value )
	{
		for (int z = 0; z < size.z; ++z)
		{
			for (int y = 0; y < size.y; ++y)
			{
				for (int x = 0; x < size.x; ++x)
				{
					data[z * (size.x * size.y) + y * (size.x) + x] = value;
				}
			}
		}
	}


	tensor_t<T> save( tensor_t<T>& input,int ch, int vert_start, int vert_end, int horiz_start, int horiz_end)
	{
		tensor_t<T> clone( *this );

		int size_w = vert_end - vert_start;
		int size_h = horiz_end - horiz_start;

	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			clone(vert_start+i,horiz_start+j,ch) = input(i,j,0);
		}
	}
		return clone;
	}

	float dot(tensor_t<T>& input) /* Dot product between x and y which has the same volume size */
	{
		float dot_out = 0.0;
		tensor_t<T> clone( *this );

		for (int i = 0; i < input.size.x ; ++i)
		{
			for (int j = 0; j < input.size.y ; ++j)
			{
				for (int ch = 0; ch < input.size.z ; ++ch)
				{
					dot_out += clone(i,j,ch) * input(i,j,ch);
				}
			}
		}
		return dot_out;
	}
	~tensor_t()
	{
		//delete[] data;

	}

};

/*
void save_tensor(tensor_t<float>& input, int ch, int vert_start, int vert_end, int horiz_start, int horiz_end)
{
	int size_w = vert_end - vert_start;
	int size_h = horiz_end - horiz_start;
	tensor_t<float> output(size_w,size_h,1);

	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			output(i,j,0)=input(vert_start+i,horiz_start+j,ch);
		}
	}
	
}*/

tensor_t<float> mul_scalar(tensor_t<float> &input, int ch, float value)
{
	int size_w = input.size.x;
	int size_h = input.size.y;
	tensor_t<float> output(size_w,size_h,1);

	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			output(i,j,0)=input(i,j,ch) * value;
		}
	}
	return output;

}
tensor_t<float> mul_tensor(tensor_t<float> &input1, tensor_t<float> &input2)
{
	int size_w = input1.size.x;
	int size_h = input2.size.y;
	int size_c = input1.size.z;
	tensor_t<float> output(size_w,size_h,size_c);

for (int c = 0; c < size_c; ++c)
{
	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			output(i,j,c) = input1(i,j,c) * input2(i,j,c);
		}
	}
}
	return output;
}

tensor_t<float> tensor_add(tensor_t<float> &input1, tensor_t<float> &input2)
{
	int size_w = input1.size.x;
	int size_h = input2.size.y;
	int size_c = input1.size.z;
	tensor_t<float> output(size_w,size_h,1);

for (int c = 0; c < size_c; ++c)
{
	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			output(i,j,c)=input1(i,j,c) + input2(i,j,c);
		}
	}
}
	return output;
}



/*Print all data in tensor*/
static void print_tensor( tensor_t<float>& data)
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
		printf( "[Dim%d]\n", z );
		for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
				printf( "%.3f \t", (float)data.get( x, y, z ) );
			}
			printf( "\n" );
		}
	}
	printf("-------------------------\n");
}
static void print_tensor_vector(std:: vector<tensor_t<float>> data)
{
	int mx = data[0].size.x;
	int my = data[0].size.y;
	int mz = data[0].size.z;

for (int vec = 0; vec < (int)data.size(); ++vec)
{
printf( "vec %d\n", vec );
	for ( int z = 0; z < mz; z++ )
	{
		printf( "[Dim%d]\n", z );
		for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
				printf( "%.3f \t", (float)data[vec].get( x, y, z ) );
			}
			printf( "\n" );
		}
	}
	printf("--------- end vec %d------------\n",vec);
}

}
//create new tensor from another tensor with specific element 
tensor_t<float> copy_tensor(tensor_t<float>& input, int ch, int vert_start, int vert_end, int horiz_start, int horiz_end)
{
	int size_w = vert_end - vert_start;
	int size_h = horiz_end - horiz_start;
	tensor_t<float> output(size_w,size_h,1);

	for (int i = 0; i < size_h; ++i)
	{
		for (int j = 0; j < size_w; ++j)
		{
			output(i,j,0)=input(vert_start+i,horiz_start+j,ch);
		}
	}
	return output;
}

tensor_t<float> create_tensor(int w, int h, int ch,float value)
{
	tensor_t<float> out (w,h,ch);

for (int i = 0; i < h; ++i)
{
	for (int j = 0; j < w; ++j)
	{
		for (int c = 0; c < ch; ++c)
		{
			out(i,j,c)= value;//1.0+(1.0+(i+j))/10;
		}
	}
}
//print_tensor(out);
return out;
}

tensor_t<float> Padding(tensor_t<float>& input, int padsize)
{
	int w = input.size.x;
	int h = input.size.y;
	int ch = input.size.z;

	int n_h = h+2*padsize;
	int n_w = w+2*padsize;

	tensor_t<float> output(n_w,n_h,ch);

	for (int i = 0; i < n_h; ++i)
	{
	for (int j = 0; j < n_w; ++j)
	{
		for (int c = 0; c < ch; ++c)
		{
			if ((i >= padsize && j >= padsize) && (i < n_h - padsize && j < n_w - padsize ))
			{
				output(i,j,c) = input(i - padsize,j - padsize,c);
			}
			else
			{
				output(i,j,c) = 0.0;
			}
		}
	}
	}
	//printf("this is output padding:\n");
	//print_tensor(output);
	return output;
}
tensor_t<float> flip_tensor(tensor_t<float> & input)
{
	tensor_t<float> output(input.size.x, input.size.y, input.size.z);

	for (int i = input.size.x - 1 ; i >=0; --i)
	{
		for (int j = input.size.y - 1 ; j >= 0; --j)
		{
			output(input.size.x - (i+1), input.size.y - (j+1),0) = input(i,j,0);
		}
	}
return output;

}
tensor_t<float> minus_tensor(tensor_t<float> intput1, tensor_t<float> intput2)
{
	tensor_t<float> output(intput1.size.x, intput1.size.y, intput1.size.z);

	for (int i = 0; i < intput1.size.z; ++i)
	{
		output(0,0,i) = (float) (intput1(0,0,i) - intput2(0,0,i));
	}
	return output;
}

float sum_tensor(tensor_t<float> input)
{
	float sum = 0;
	for (int w = 0; w < input.size.x; ++w)
	{
		for (int h = 0; h < input.size.y; ++h)
		{
			for (int c = 0; c < input.size.z; ++c)
			{
				sum += input(w,h,c);
			}
		}
	}
}