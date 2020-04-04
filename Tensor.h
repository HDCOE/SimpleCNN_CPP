
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

		for (int i = 0; i < _x * _y * _z; ++i)
		{
			this->data[i] = i+1; //0; 
		}

	}
	tensor_t( const tensor_t& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(this->data, other.data, other.size.x *other.size.y *other.size.z * sizeof( T ));
		this->size = other.size;
	}
/* Define element of tensor by tensor(x,y,z)= value */
	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z ) // swap y ,x
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
		clone.clear();
		return dot_out;
	}
	~tensor_t()
	{
		//delete[] data;
	}
	tensor_t<T> clear()
	{
		delete[] data;
	}

};

/*Print all data in tensor*/
static void print_tensor( tensor_t<float>& data)
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	printf("dimension %d x %d x %d \n", mx, my, mz );
	for ( int z = 0; z < mz; z++ )
	{
		printf("-----------------------------\n");//printf( "[Dim%d]\n", z );
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
	output.clear();
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

double unifRand()
{
    return rand() / double(RAND_MAX);
}
//
// Generate a random number in a real interval.
// param a one end point of the interval
// param b the other end of the interval
// return a inform rand numberin [a,b].
double unifRand(double a, double b)
{
    return (b-a)*unifRand() + a;
}

int index_max(tensor_t<float> input)
{
	float max = input(0,0,0);
	float idx_max = 0;

	for (int i = 0; i < input.size.z ; ++i)
	{
		if (max < input(0,0,i))
		{
			max = input(0,0,i);
			idx_max = i;
		}
	}
	return idx_max;
}

