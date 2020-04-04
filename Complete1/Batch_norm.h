
struct bash_norm_layer
{
	float eps = 0;
	tensor_t<float> x;
	tensor_t<float> x_hat; // normalize
	tensor_t<float> y;

	std::vector<float> mu;
	std::vector<float> variance;
	std::vector<float> gamma;
	std::vector<float> beta;

	// gradient 
	std::vector<float> dgamma;
	std::vector<float> dbeta;
	tensor_t<float> dx;
	int m = x.size.x * x.size.y * x.size.z;

	bash_norm_layer( float eps, point_t in_size):
	x 	 ( in_size.x, in_size.y, in_size.z),
	x_hat( in_size.x, in_size.y, in_size.z),
	y 	 ( in_size.x, in_size.y, in_size.z),
	dx 	 ( in_size.x, in_size.y, in_size.z),

	mu(in_size.z),
	variance(in_size.z),
	gamma (in_size.z),
	beta (in_size.z)
	{
		//this->gamma = gamma;
		//this->beta = beta;
		this->eps = eps;

	// initialize gamma, beta
		for (int i = 0; i < (int)gamma.size(); ++i)
		{
			gamma[i] = 0.1;
			beta [i] = 0;
		}
	}

void forward_batchnorm()
{
	/* 1 mean, 2 variance , 3 normalize, 4 shift and scale*/
	float mu ,variance = 0;
	float m = (float) x.size.x * x.size.y;
	float sum = 0;

// 1 mean
	for (int c = 0; c < x.size.z; ++c)
	{
		for (int w = 0; w < x.size.x; ++w)
		{
			for (int h = 0; h < x.size.y; ++h)
			{
				sum += x(w,h,c);
			}
		}
		mu = (1.0f / m) * sum;
		this->mu[c] = mu; // mean
		sum = 0;

	}	

// 2 variance
	for (int c = 0; c < x.size.z; ++c)
		{
		for (int w = 0; w < x.size.x; ++w)
		{
			for (int h = 0; h < x.size.y; ++h)
			{
				variance += pow((x(w,h,c) - this->mu[c]),2);
				//printf("this is variance :%f - %f this is m:%f\n",x(w,h,c), this->mu[c],m);
			}
		}
		variance = (1.0f / m) * variance;
		this->variance[c] = variance;
		
		variance = 0;

	}
	

// 3 normalize
	for (int c = 0; c < x.size.z; ++c)
		{
		for (int w = 0; w < x.size.x; ++w)
		{
			for (int h = 0; h < x.size.y; ++h)
			{
				x_hat(w,h,c) = (x(w,h,c) - this->mu[c]) / sqrt(this->variance[c] + eps);
			//	printf("this is x_hat:%f\n",sqrt(this->variance[c] + eps) );
				y(w,h,c) = gamma[c] * x_hat(w,h,c) + beta[c];
			}
		}
	}

}
/*
 	dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # for dx visit this backprop diagram:
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    dx_hat = dout * gamma
    dxmu1 = dx_hat * 1 / np.sqrt(var + eps)
    divar = np.sum(dx_hat * (x - mu), axis=0)
    dvar = divar * -1 / 2 * (var + eps) ** (-3/2)

    dsq = 1 / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * (x - mu) * dsq
    dx1 = dxmu1 + dxmu2
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
    dx2 = 1 / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2
*/

void backward_batchnorm(tensor_t<float> dout)
{

   std::vector<float> dbeta(dout.size.z);// sum_tensor(dout);// np.sum(dout, axis=0)
   std::vector<float> dgamma(dout.size.z); //  np.sum(dout * x_hat, axis=0)
   std::vector<float> dvariance(dout.size.z);
   std::vector<float> dmu(dout.size.z);

    tensor_t<float> dsq(x.size.x, x.size.y, x.size.z);
    tensor_t<float> dx1(x.size.x, x.size.y, x.size.z);
    tensor_t<float> dx2(x.size.x, x.size.y, x.size.z);

    dsq = create_tensor(x.size.x, x.size.y, x.size.z, 1);
    dx2 = create_tensor(x.size.x, x.size.y, x.size.z, 1);

	float dxmu1, dxmu2, dx_hat;

	for (int c = 0; c < x.size.z; ++c)
	{
		//initial dbeta
			dbeta[c] = 0;
			dgamma[c] = 0;
			dvariance[c] = 0;
		
		for (int w = 0; w < x.size.x; ++w)
			{
				for (int h = 0; h < x.size.y; ++h)
				{
					dbeta[c] += dout(w,h,c); //dbeta = sum(dout)
					dgamma[c] += dout(w,h,c) * x_hat (w,h,c); // dgamma = sum( dout * x_hat)			
					
					dx_hat = dout(w,h,c) * gamma[c];  // dx_hat = dout * gamma

				//	dxmu1 = dx_hat * 1 / (sqrt(variance+eps));
					dvariance[c] += dx_hat * (x(w,h,c) - mu[c]);
				}
			}
		dvariance[c] = dvariance[c] *  -1.0f / sqrt(variance[c]+eps);
		//printf("this is dbeta%f \n",dbeta[c] );
	}
		
	for (int c = 0; c < x.size.z; ++c)
		{
			dmu[c] = 0;
			for (int w = 0; w < x.size.x; ++w)
				{
					for (int h = 0; h < x.size.y; ++h)
					{
				
						dx_hat = dout(w,h,c) * gamma[c];  // dx_hat = dout * gamma
						dxmu1 = dx_hat * 1.0f / (sqrt(variance[c]+eps));

						dsq(w,h,c) =  (1.0f / m) * dsq(w,h,c) * dvariance[c]  ; 
					  
						dxmu2 = 2.0f * (x(w,h,c) - mu[c]) * dsq(w,h,c);

						dx1(w,h,c) = dxmu1 + dxmu2;
						dmu[c] += dx1(w,h,c);
				}
			}
		}
	
	for (int c = 0; c < x.size.z; ++c)
		{
			for (int w = 0; w < x.size.x; ++w)
			{
				for (int h = 0; h < x.size.y; ++h)
				{
					dx2(w,h,c) = (1.0f / m) * dx2(w,h,c) * (-dmu[c]);

					dx(w,h,c) = dx1(w,h,c) + dx2(w,h,c);
				}
			}
		}
		//printf("this is dx1\n"); print_tensor(dx1);
		this->dbeta = dbeta;
		this->dgamma = dgamma;
}

void batchnorm_updata_gradient()
{
	for (int i = 0; i < (int)gamma.size(); ++i)
	{
		gamma[i] = gamma[i] - 0.1 * dgamma[i];
		beta[i] = beta[i] - 0.1 * dbeta[i];
	}
	
}

};
