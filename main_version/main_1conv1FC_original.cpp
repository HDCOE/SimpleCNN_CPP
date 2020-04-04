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
#include "Param.h"

#include "CONV_layer.h"

#include "POOLING_layer.h"
#include "RELU_layer.h"
#include "FC_layer.h"

#include "Generate_data.h"
#include "Batch_norm.h"
using namespace std;
/*struct dataset
{
	tensor_t<float> data;
	tensor_t<float> y_out;
};
void Dataset_and(vector<dataset>& cases)
{
	
	dataset set {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
//	vector<dataset> cases;
	// ab
	set.data(0,0,0) = 0;
	set.data(0,1,0) = 0;
    // cd
	set.data(1,0,0) = 0;
	set.data(1,1,0) = 0;

	set.y_out(0,0,0) = 0; //y1 = ab
	set.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set);

dataset set1 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set1.data(0,0,0) = 0;
	set1.data(0,1,0) = 1;
    // cd
	set1.data(1,0,0) = 0;
	set1.data(1,1,0) = 1;

	set1.y_out(0,0,0) = 0; //y1 = ab
	set1.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set1);

dataset set2 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set2.data(0,0,0) = 1;
	set2.data(0,1,0) = 0;
    // cd
	set2.data(1,0,0) = 1;
	set2.data(1,1,0) = 0;

	set2.y_out(0,0,0) = 1; //y1 = ab
	set2.y_out(0,0,1) = 0; //y2 = ab
	cases.push_back(set2);

dataset set3 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set3.data(0,0,0) = 1;
	set3.data(0,1,0) = 1;
    // cd
	set3.data(1,0,0) = 1;
	set3.data(1,1,0) = 1;

	set3.y_out(0,0,0) = 0; //y1 = ab
	set3.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set3);
}
*/
int main ()
{

	int pad =2;
	int stride =1;
	int Wsize = 2;
	int Nfilter = 2;

	float error = 0, error_total = 0.0;

	vector<dataset> dataset_and; 
	dataset_and =  Dataset_and();
	tensor_t<float> input(2,2,1);

// initialize y_hat
	std::vector<tensor_t<float>> y_hat_vec;
    tensor_t<float> y_hat(dataset_and[0].y_out.size.x, dataset_and[0].y_out.size.y, dataset_and[0].y_out.size.z);

	for (int i = 0; i < (int)dataset_and.size(); ++i)
	{
		y_hat = create_tensor(dataset_and[0].y_out.size.x, dataset_and[0].y_out.size.y, dataset_and[0].y_out.size.z, 0);
		y_hat_vec.push_back(y_hat);
		print_tensor(dataset_and[2].data);
	}

// layer conv1
	conv_layer conv1(Wsize, Nfilter, stride, pad, input.size );
// layer conv2
	conv_layer conv2(Wsize, Nfilter, stride, pad-1, conv1.output.size );
// layer fc
	fc_layer fc1(2, conv2.output.size);
// layer fc
	int Nclass = 2;
	fc_layer fc_out(Nclass, fc1.output.size );

//gradient dE, dZ (sigmoid)
	tensor_t<float> dE(1,1, fc_out.output.size.z);
	tensor_t<float> dZ(1,1, fc_out.output.size.z);

//start training	
	for (int epoch = 0; epoch < 4; ++epoch)
	{
		error_total = 0;
		for (int i = 2; i < 3 ; ++i)
		{
		
	// layer conv1
			conv1.input =  dataset_and[i].data; // input_all[i];
			conv1.Conv_forward();

	// layer conv2
			conv2.input = conv1.output;
			conv2.Conv_forward();
			print_tensor(conv2.output);

	// layer fc1
			fc1.input =  conv2.output;
			fc1.forward_fc();


			
 	// layer fc out
			fc_out.input = fc1.output;
			fc_out.forward_fc();
			//BNorm.x = fc_out.output;
			//BNorm.forward_batchnorm();


	// layer softmax
			y_hat_vec[i] = Softmax(fc_out.output); //y_hat[i] = forward_Sigmoid(fc1.output);		

			error_total += Cross_entropy(y_hat_vec[i], dataset_and[i].y_out);



///////////////////  bacward //////////////////////////
			dZ = Back_Softmax(y_hat_vec[i], dataset_and[i].y_out);


	// backward fc out
			//BNorm.backward_batchnorm(dZ);
			fc_out.backward_fc(dZ);	

	// backward fc1
			fc1.backward_fc(fc_out.gradient.dA);//(relu1.gradient_dA);

	// backward batch norm
			//BNorm.backward_batchnorm(fc1.gradient.dA);

	//backward conv2
			conv2.dZ = fc1.gradient.dA;
			conv2.Conv_backward();

	// bacward conv1
			conv1.dZ = conv2.gradient.dA;
			conv1.Conv_backward(); //conv1.gradient = Conv_backward(relu1.gradient_dA, conv1.input, conv1.W, pad);

			print_tensor(conv1.gradient.dA);
	
	// update weight fc
			fc_out.fc_weight_update();
			//BNorm.batchnorm_updata_gradient();
			fc1.fc_weight_update();
			
	// update weight conv
			conv2.conv_weight_update();
			conv1.conv_weight_update();

		//	printf("this is fc out output\n"); print_tensor(fc_out.output);
		}
	printf("this is epoch :%d, error : %.5f \n", epoch, error_total);
	}
	printf("this is sigmoid \n"); print_tensor_vector(y_hat_vec);
	printf("this is fc_out dA \n"); print_tensor(fc_out.gradient.dA);
	printf("this is fc1 W\n"); print_tensor_vector(fc1.W);
	printf("this is fc1 out\n"); print_tensor(fc1.output);
}
/*
printf("this is conv1 input\n"); print_tensor(conv1.input);
printf("this is conv1 W\n"); print_tensor_vector(conv1.W);
printf("this is conv1 output\n"); print_tensor(conv1.output);
printf("this is relu1 output\n"); print_tensor(relu1.output);
printf("this is pooling1 output\n"); print_tensor(pooling1.output);
printf("this is fc1 W\n"); print_tensor_vector(fc1.W);
printf("this is fc1 output\n"); print_tensor(fc1.output);
printf("this is sigmoid output\n"); print_tensor(y_hat_vec[i]);

printf("this is gradient E respect to y_hat\n"); print_tensor(dE);
printf("this is fc1:dZ\n"); print_tensor(dZ);
printf("this is fc1:dW\n"); print_tensor_vector(fc1.gradient.dW);
printf("this is fc1:dA\n"); print_tensor(fc1.gradient.dA);
printf("this is pooling1:dA\n"); print_tensor(pooling1.gradient_dA);
printf("this is relu1:dA \n"); print_tensor(relu1.gradient_dA);
printf("this is conv1:dW \n"); print_tensor_vector(conv1.gradient.dW);
printf("this is conv1:dA \n"); print_tensor(conv1.gradient.dA);

error = 0;
	int m = (int) dataset_and.size();
	for (int filter = 0; filter < m; ++filter)
	{
		for (int i = 0; i < dataset_and[0].y_out.size.z ; ++i)
		{
			//error += 0.5 * pow((dataset_and[filter].y_out(0,0,i) - y_hat_vec[filter](0,0,i)), 2) ;
		}
	}
	//error = error / m;
	//printf("this is epoch :%d, error : %.5f \n", epoch, error);
*/