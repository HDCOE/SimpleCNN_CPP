
vector<tensor_t<float>> weight_update(vector<tensor_t<float>> W, vector<tensor_t<float>> dW ) //simple SGC gradient
{
	float learning_rate = 0.05;
	int n_h = W[0].size.x;
	int n_w = W[0].size.y;
	int n_c = W[0].size.z;

	vector<tensor_t<float>> out;

	for (int filter = 0; filter < (int)W.size(); ++filter)
	{
		tensor_t<float> temp(n_h,n_w,n_c);

		for (int i = 0; i < n_h; ++i)
		{
			for (int j = 0; j < n_w; ++j)
			{
				for (int ch = 0; ch < n_c; ++ch)
				{
					temp(i,j,ch) = W[filter](i,j,ch) - learning_rate * dW[filter](i,j,ch);
				}
			}
		}
		out.push_back(temp);
	}

return out;
}

tensor_t<float> minus_tensor(tensor_t<float> intput1, tensor_t<float> intput2)
{
	tensor_t<float> output(intput1.size.x, intput1.size.y, intput1.size.z);

	for (int i = 0; i < intput1.size.z; ++i)
	{
		output(0,0,i) = intput1(0,0,i) - intput2(0,0,i);
		//printf("out1 %f - out2 %.3f\n",intput1(0,0,i), intput2(0,0,i) );
	}
	return output;
}
