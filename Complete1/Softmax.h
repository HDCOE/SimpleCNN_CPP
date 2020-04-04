
/*
	sigmoid = 1 / 1 + e(-x)
*/

tensor_t<float> forward_Sigmoid( tensor_t<float>Z)
{
	
	tensor_t<float> out(Z.size.x, Z.size.y, Z.size.z);

	for (int filter = 0; filter < Z.size.z ; ++filter)
	{
		out(0,0,filter) = 1.0f / (1.0f + exp(-Z(0,0,filter)));
	
	/*	if (out(0,0,filter)>0.5)
		{
			out(0,0,filter) =1;
		}
		else
			out(0,0,filter)=0;
	*/
	}
	return out;
}
/*
 derivative of sigmoid y = 1 / 1 + e(-x) then dy = y * (1-y)
*/
tensor_t<float> backward_Sigmoid( tensor_t<float>Z, tensor_t<float>dE)
{
	
	tensor_t<float> out(Z.size.x, Z.size.y, Z.size.z);

	float y =0.0f;

	for (int filter = 0; filter < Z.size.z ; ++filter)
	{
		y =  Z(0,0,filter);
		out(0,0,filter) = (y * (1 - y)) * dE(0,0,filter);
	}
	return out;
}

tensor_t<float> Softmax( tensor_t<float>Z)
{
	
	tensor_t<float> out(Z.size.x, Z.size.y, Z.size.z);

	float exp_z =0;
	float sum_exp_z =0;
	float max_value =  max(Z);
	float x;

	for (int filter = 0; filter < Z.size.z ; ++filter)
	{
		x = Z(0,0,filter) - max_value; // Numeric stability 
		sum_exp_z += exp(x);
		//printf("this is Z:%f ,exp[%d]:%f\n",Z(0,0,filter),filter, exp(Z(0,0,filter)));
	
	}


	for (int filter = 0; filter < Z.size.z ; ++filter)
	{
		x = Z(0,0,filter) - max_value;
		exp_z = exp(x);
		out(0,0,filter) = exp_z / sum_exp_z;
	}
	return out;
}

/* E = (-1/n) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
	dZ = -1*(y/y_hat)+(1-y)/(1-y_hat)

*/

tensor_t<float> Back_Softmax( tensor_t<float> y_hat, tensor_t<float> y) // dZ = y_hat - y gradient of L respect to input softmax
{
	tensor_t<float> dA(y.size.x, y.size.y, y.size.z);
	float E = 0.0;
	tensor_t<float> dZ(y.size.x, y.size.y, y.size.z);

	float n = (float) y.size.z;

	
	for (int filter = 0; filter < y.size.z ; ++filter)
	{
		// dZ:dE/dz
		//dZ(0,0,filter) = -1 * ((y(0,0,filter) / y_hat(0,0,filter)) + (1-y(0,0,filter)) / (1-y_hat(0,0,filter)));

		//dA: dE/da  dE/dz*dz/da, dz/da = z(1-z), z = y_hat
		//dA(0,0,filter) = dZ(0,0,filter) * y_hat(0,0,filter) * (1-y_hat(0,0,filter));
		dA(0,0,filter) = y_hat(0,0,filter) - y(0,0,filter);

	}
	return dA;
}

float Cross_entropy(tensor_t<float> y_hat, tensor_t<float> y) // softmax cross entropy: E = (-1/n) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
{
	float E = 0.0;
	float n = (float) y.size.z;
	for (int filter = 0; filter < y.size.z ; ++filter)
	{

		E += (y(0,0,filter) * log(y_hat(0,0,filter))) + (1-y(0,0,filter))*log(1-y_hat(0,0,filter));
		E = (-1/n) * E;
	}
	return E;
}



/* Everything is same as backward_fc except dZ is changed */
// y hat is forward prop output

cache backward_Softmax(tensor_t<float>y_hat, tensor_t<float>y, tensor_t<float> a_prev, vector<tensor_t<float>> W, tensor_t<float> bias)
{
	/*
	def softmax_backward(y_pred, y, w, b, x):
    	m = y.shape[1]
    	dZ = y_pred - y
    	dW = (1/m)*dZ.dot(x.T)
    	db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    	dx =  np.dot(w.T,dZ)
    return dx, dW,db
    */
    float m = (float) y.size.z;

	int n_h = a_prev.size.x ; int n_w = a_prev.size.y; int n_c = a_prev.size.z;

	
	tensor_t<float> dZ(y_hat.size.x, y_hat.size.y, y_hat.size.z); // dZ with same size as Z
	tensor_t<float> dA_prev(a_prev.size.x, a_prev.size.y, a_prev.size.z); // dA with same size as a_prev
	vector<tensor_t<float>> dW;
	tensor_t<float> db(1,1,(int)W.size());
	db = create_tensor(1,1,db.size.z,0);

// calculate dZ = dA*relu(dZ)
	for (int i = 0; i < dZ.size.z; ++i)
	{
		dZ(0,0,i) = y_hat(0,0,i) - y(0,0,i);
	}

//initialize vector dW
	for (int i = 0; i < (int)W.size(); ++i)
	{
		dW.push_back(create_tensor(W[0].size.x,W[0].size.y,W[0].size.z,0));
	}
//initialize dA
	dA_prev = create_tensor(dA_prev.size.x, dA_prev.size.y, dA_prev.size.z,0);

	for (int filter = 0; filter < (int)W.size(); ++filter)
	{
		for (int i = 0; i < n_h; ++i)
		{
			for (int j = 0; j < n_w; ++j)
			{
				for (int ch = 0; ch < n_c; ++ch)
				{
					// dW = 1/m *dZ.dot(a_prev)
					dW[filter](i,j,ch) = (1/m) * dZ(0,0,filter)*a_prev(i,j,ch);

					// dA = 1/m * WT.dot(dZ)
					dA_prev(i,j,ch) += (1/m) * W[filter](i,j,ch) * dZ(0,0,filter);
				}
			}
		}
		//calculate db
		db(0,0,filter) += (1/m) * dZ(0,0,filter);
	}
print_tensor(dA_prev);
printf("this is db"); print_tensor(db);

cache return_cache(dA_prev.size ,dW[0].size, (int)dW.size(), db.size);
return_cache.dA = dA_prev;
return_cache.dW = dW;
return_cache.db = db;
return return_cache;
}