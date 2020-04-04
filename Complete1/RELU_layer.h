struct relu_layer
{
	tensor_t<float> input;
	tensor_t<float> output;
	tensor_t<float> gradient_dA;

	relu_layer(point_t size):
	input(size.x, size.y, size.z),
	output(size.x, size.y, size.z),
	gradient_dA (size.x, size.y, size.z)
	{

	}

void Forward_ReLu()
{
	for (int i = 0; i < input.size.x; ++i)
	{
		for (int j = 0; j < input.size.y; ++j)
		{
			for (int ch = 0; ch < input.size.z; ++ch)
			{
				if(input(i,j,ch) < 0)
					output(i,j,ch) = 0;
				else
					output(i,j,ch) = input(i,j,ch);
			}
		}
	}

}

/* Backward ReLu output is: 0 when x <= 0, 1 when x > 0*/
void Backward_ReLu(tensor_t<float>& dZ) 
{
	for (int i = 0; i < dZ.size.x; ++i)
	{
		for (int j = 0; j < dZ.size.y; ++j)
		{
			for (int ch = 0; ch < dZ.size.z; ++ch)
			{
				if(input(i,j,ch) <= 0)
					gradient_dA(i,j,ch) = 0;
				else
					gradient_dA(i,j,ch) = 1.0 * dZ(i,j,ch);
			}
		}
	}
}

/*
void Backward_ReLu(tensor_t<float>& dZ) 
{
	for (int i = 0; i < dZ.size.x; ++i)
	{
		for (int j = 0; j < dZ.size.y; ++j)
		{
			for (int ch = 0; ch < dZ.size.z; ++ch)
			{
				if(dZ(i,j,ch) <= 0)
					gradient_dA(i,j,ch) = 0;
				else
					gradient_dA(i,j,ch) = 1;
			}
		}
	}
}
*/
};




