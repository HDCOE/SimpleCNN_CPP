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
//#include "Weightupdate.h"
#include "Generate_data.h"
#include "Batch_norm.h"
using namespace std;

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}
struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};

vector<case_t> read_test_cases()
{
	vector<case_t> cases;

	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < case_count; i++ )
	{
		case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 1, 1, 10 )};//case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			c.out( 0, 0, b ) = *label == b ? 1.0f : 0.0f; //c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main ()
{

	int pad =1;
	int stride =2;
	int Wsize = 6;
	int Nfilter = 2;

	float error = 0.0;

// generate ab + cd 
//input 00 + 00 = 0

 	vector<case_t> dataset = read_test_cases();

    std::vector<tensor_t<float>> y_hat_vec;
    int size_x = dataset[0].out.size.x; int size_y = dataset[0].out.size.y ; int size_z = dataset[0].out.size.z;

    tensor_t<float> y_hat(size_x, size_y, size_z );

	for (int i = 0; i < (int)dataset.size(); ++i)
	{
		y_hat = create_tensor(size_x, size_y, size_z, 0);
		y_hat_vec.push_back(y_hat);
	}
// layer conv1
	conv_layer conv1(Wsize, Nfilter, stride, pad, dataset[0].data.size);

// batch norm
	//bash_norm_layer BNorm( 0.01, conv1.output.size); 
// Relu 1
	relu_layer relu1(conv1.output.size);
// pooling 1
	int poolsize = 4; int poolstride = 4;
	int mode = 1;// max :0, average:1
	pool_layer pooling1(poolsize, poolstride, mode, conv1.output.size);

// layer fc
	int Nclass = 10;
	fc_layer fc1(Nclass, pooling1.output.size);

//gradient Z
	tensor_t<float>  dE(1,1, fc1.output.size.z);
	tensor_t<float> dZ(1,1, fc1.output.size.z);


//start training

	for (int epoch = 0; epoch < 1000 ;++epoch)
	{
		error = 0;
		for (int i = 0; i < 1 ; ++i)
		{
			conv1.input = dataset[i].data;

	// layer conv1
			conv1.Conv_forward();

			//BNorm.x = conv1.output;
			//BNorm.forward_batchnorm();
			
	// layer relu
			relu1.input = conv1.output;
			relu1.Forward_ReLu();		

	// layer pooling
			pooling1.input = relu1.output;
			pooling1.forward_pooling(); //input, poolsize, stride, mode

	// layer fc
			fc1.input = pooling1.output;
			fc1.forward_fc();

	// layer sigmoid
			y_hat_vec[i] = Softmax(fc1.output);	

			error += Cross_entropy(y_hat_vec[i], dataset[i].out);	

///////////////////  bacward //////////////////////////
	
	// cal gradient Z from error = sum(1/2 * (y - yhat)^2) then dE/dyhat = yhat - y 
			//dE = minus_tensor(y_hat_vec[i], dataset[i].out);
		
	// bacward sigmoid
			dZ = Back_Softmax(y_hat_vec[i], dataset[i].out);//backward_Sigmoid(y_hat_vec[i], dE);

	// backward fc
		
			fc1.backward_fc(dZ);
			
	// bacward pooling
			pooling1.Backward_pooling(fc1.gradient.dA);

	// bacward relu
			relu1.Backward_ReLu(pooling1.gradient_dA);		

	// backward batch norm
			//BNorm.backward_batchnorm(relu1.gradient_dA);

	// backward conv
			conv1.dZ = relu1.gradient_dA;
			conv1.Conv_backward();
			fc1.fc_weight_update();

		}

		printf("this is epoch :%d, error %.5f \n", epoch, error);

	}
	printf("this is weight conv1\n"); print_tensor_vector(conv1.W);
	printf("this is fc1.output\n"); print_tensor(fc1.output);

}
