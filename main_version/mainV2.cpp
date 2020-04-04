#include <cassert>
#include <vector>

#include <cstdint>
#include <iostream>
#include <fstream>

#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <thread>


#include "byteswap.h"

#include "Tensor.h"
#include "Param.h"

#include "Optimization_method.h"

#include "CONV_layer.h"

#include "POOLING_layer.h"
#include "RELU_layer.h"
#include "FC_layer.h"
//#include "Weightupdate.h"
#include "Generate_data.h"
#include "Batch_norm.h"
// add progress and processing time display
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include "layer.h"

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

vector<case_t> read_train_cases()
{
	vector<case_t> cases;

	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < (int)case_count; i++ )
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

vector<case_t> read_test_cases()
{
	vector<case_t> cases;

	uint8_t* train_image = read_file( "t10k-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "t10k-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < (int)case_count; i++ )
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
void forward(layer * nn, tensor_t<float>& input)
{
	switch (nn->type)
		{
			case conv:		((conv_layer*) nn)->forward_conv(input); 	break;
			case pooling :	((pool_layer*) nn)->forward_pooling(input); break;
			case fc: 		((fc_layer*) nn)->forward_fc(input); 		break;
			case relu: 		((relu_layer*) nn)-> forward_relu(input); 	break;
		}
}

void backward(layer * nn, tensor_t<float>& gradient)
{
	switch (nn->type)
		{
			case conv:		((conv_layer*) nn)->backward_conv(gradient); 	break;
			case pooling :	((pool_layer*) nn)->backward_pooling(gradient); break;
			case fc: 		((fc_layer*) nn)->backward_fc(gradient); 		break;
			case relu: 		((relu_layer*) nn)-> backward_relu(gradient); 	break;
		}
}

void update_weight(layer * nn)
{
	switch (nn->type)
		{
			case conv:		((conv_layer*) nn)->conv_weight_update(); 	break;
			case pooling :	break;
			case fc: 		((fc_layer*) nn)->fc_weight_update(); 		break;
			case relu: 		break;
		}
}

float train_model(vector<layer*>& layers, tensor_t<float>& data, tensor_t<float>&  y)
{
	float error;
	int layers_size = (int)layers.size();
	int final_layer = layers_size - 1;
	// forward calculation
	classifier * estimate = new classifier(y.size);
	// y hat
	tensor_t<float> y_hat(y.size.x, y.size.y , y.size.z);
	//gradient Z
	tensor_t<float> dZ(1,1, y.size.z);

		// forward 
		forward(layers[0], data);
		for (int i = 1; i < layers_size; i++)
		{
			forward( layers[i], layers[i - 1]->output);
		}
		// classification
		estimate->forward_Softmax(layers[final_layer]->output);
		error += Cross_entropy(estimate->y_hat, y);
        
        //backward
        estimate->backward_softmax(y);
		backward(layers[final_layer], estimate->gradient_dA);
		
		for (int i = layers_size; i <= 0; i--)
		{
			backward( layers[i -1 ], layers[i]->gradient_dA);
		}

		// update weight
		for (int i = 0; i < layers_size; i++)
		{
			update_weight(layers[i]);
		}
}

int main(int argc, char const *argv[])
{
	tensor_t<float> test(4,4,3);
	tensor_t<float> y(1,1,5);
	y.set_all(0);
	test.set_all(1);
	test(0,0,0) = 1.2;
	test(3,3,2) = 0.8;
	

	vector<layer*> layers;

	pool_layer * s2 = new pool_layer(2, 2, 0, test.size );
	conv_layer * c1 = new conv_layer(2, 2, 1, 2, s2->output.size); //Wsize, Nfilter, stride, pad, dataset[0].data.size
	relu_layer * c1_relu = new relu_layer(c1->output.size);

	fc_layer * fc1 = new fc_layer(5, c1->output.size);
	fc_layer * fc2 = new fc_layer(5, fc1->output.size);

	layers.push_back( (layer*) s2 );
	layers.push_back( (layer*) c1);
	layers.push_back( (layer*) c1_relu);
	layers.push_back( (layer*) fc1);
	layers.push_back( (layer*) fc2);

	float error = train_model(layers, test, y);

	print_tensor(c1_relu->output);
	print_tensor(layers[2]->gradient_dA);

	return 0;
}
int main_()
{
	int pad =1;
	int stride =2;
	int Wsize = 6;
	int Nfilter = 2;

	float error = 0.0;

 	vector<case_t> dataset = read_train_cases();
 	vector<case_t> testset = read_test_cases();

    std::vector<tensor_t<float>> y_hat_vec, y_hat_vec_test;
    int size_x = dataset[0].out.size.x; int size_y = dataset[0].out.size.y ; int size_z = dataset[0].out.size.z;

    tensor_t<float> y_hat(size_x, size_y, size_z );
// set y hat = 0
	for (int i = 0; i < (int)dataset.size(); ++i)
	{
		y_hat = create_tensor(size_x, size_y, size_z, 0);
		y_hat_vec.push_back(y_hat);
	}

// set y hat test = 0
	tensor_t<float> y_hat_test(testset[0].out.size.x, testset[0].out.size.y, testset[0].out.size.z );
	for (int i = 0; i < (int)testset.size(); ++i)
	{
		y_hat_test = create_tensor(testset[0].out.size.x, testset[0].out.size.y, testset[0].out.size.z, 0);
		y_hat_vec_test.push_back(y_hat_test);
	}

// TEST LENET -5
// layer conv1 5x5 , 6 filters, stride 1, pad 0
	conv_layer * c1 = new conv_layer(5, 6, 1, 0, dataset[0].data.size); //Wsize, Nfilter, stride, pad, dataset[0].data.size
	relu_layer * c1_relu = new relu_layer(c1->output.size);

// pooling 1 2x2, stride 2 // max :0, average:1
	pool_layer * s2 = new pool_layer(2, 2, 1, c1_relu->output.size ); //(poolsize, poolstride, mode, conv1->output.size);

// conv2 5x5, 16 filters, stride 1, pad 0
	conv_layer * c3 = new conv_layer(5, 16, 1, 0, s2->output.size);
	relu_layer * c3_relu = new relu_layer(c3->output.size);

// pooling 2 2x2, stride 2 , mode 0 (max)
	pool_layer * s4 = new pool_layer(2, 2, 1, c3_relu->output.size);

// conv3 5x5, 120 filters, stride 1, pad 0 (now 4x4 bcoz : input change from 32x32 to 28x28)
	conv_layer * c5 = new conv_layer(4, 120, 1, 0, s4->output.size);
	relu_layer * c5_relu = new relu_layer(c5->output.size);

// FC layer
	//fc_layer * fc6 = new fc_layer(84, c5_relu->output.size); // (Nclass, inputsize)
//classifier
	fc_layer * fc_out = new fc_layer(10, c5_relu->output.size);
	classifier * estimate = new classifier(fc_out->output.size);

//gradient Z
	tensor_t<float>  dE(1,1, fc_out->output.size.z);
	tensor_t<float> dZ(1,1, fc_out->output.size.z);


//start training
float n  = 0;
int testing_size =  (int)testset.size(); // 100
int training_size = (int)dataset.size();//(int)dataset.size();// 2000
int mini_batch = 60; //16;

// Thread add

// Display
 boost::progress_display disp1(training_size);
 boost::timer t;
	for (int epoch = 0; epoch < 10 ;++epoch)
	{			
		error = 0.0;
		n = 0;
		
		for (int k = 0; k < training_size ; k += mini_batch)  // for (int k = 0; k < training_size / mini_batch ; ++k) 
		{	
		 
		 for (int i = k ; i < mini_batch + k ; ++i) //for (int i = mini_batch * k; i < mini_batch * k + mini_batch ; ++i)
			{
				n++;
	// Conv c1
			    c1->forward_conv(dataset[i].data); 
				c1_relu->forward_relu(c1->output);
	// Pool s2
				s2->forward_pooling(c1_relu->output);
	// Conv c3
			    c3->forward_conv(s2->output);
				c3_relu->forward_relu(c3->output);
	// Pool s4
				s4->forward_pooling(c3_relu->output);
	// Conv c5
				c5->forward_conv(s4->output);
				c5_relu->forward_relu(c5->output);
	// FC 
				//fc6->forward_fc(c5_relu->output);

	// classifire
				fc_out->forward_fc(c5_relu->output);

				estimate->forward_Softmax(fc_out->output);
				y_hat_vec[i] = estimate->y_hat; 	

				error += Cross_entropy(y_hat_vec[i], dataset[i].out);


	///////////////////  bacward /////////////////////////		

				estimate->backward_softmax(dataset[i].out);
				dZ = estimate->gradient_dA;

				fc_out->backward_fc(dZ);

	// FC
				//fc6->backward_fc(fc_out->gradient.dA);
	// conv c5
				c5_relu->backward_relu(fc_out->gradient_dA);
			    c5->backward_conv(c5_relu->gradient_dA);  
	// Pool s4
			    s4->backward_pooling(c5->gradient_dA);
	// Conv c3
			    c3_relu->backward_relu(s4->gradient_dA);
			    c3->backward_conv(c3_relu->gradient_dA);
	// Pool s2
			    s2->backward_pooling(c3->gradient_dA);
	// Conv c1
			    c1_relu->backward_relu(s2->gradient_dA);
			    c1->backward_conv(c1_relu->gradient_dA);
		
		}	
		
	// update weights 
			    fc_out->fc_weight_update() ;// fc_out->fc_weight_update();
				//fc6->fc_weight_update();
				c5->conv_weight_update();
				c3->conv_weight_update();
				c1->conv_weight_update();

	// display update	
 		 disp1 += mini_batch;	
 		//cout << "Train " << training_size << ": this is epoch: " << epoch << " error (%) = " << ((-1 / n) * error)  << endl;

		}
		// display processing time and error
		std::cout << endl << t.elapsed() << "s elapsed." << std::endl;
		cout << "Train " << training_size << ": this is epoch: " << epoch << " error (%) = " << ((-1 / n) * error)  << endl;
		// restart timer
		t.restart();

////////////////////////////////////////////////////////////
// Test model
////////////////////////////////////////////////////////////
		 error = 0.0; n = 0;
		for (int m = 0; m < testing_size; ++m)
		{
			n++;
			// Conv c1
				c1->forward_conv(testset[m].data); 
				c1_relu->forward_relu(c1->output);
	// Pool s2
				s2->forward_pooling(c1_relu->output);
	// Conv c3
				c3->forward_conv(s2->output);
				c3_relu->forward_relu(c3->output);
	// Pool s4
				s4->forward_pooling(c3_relu->output);
	// Conv c5
				c5->forward_conv(s4->output);
				c5_relu->forward_relu(c5->output);
	// FC 
				//fc6->forward_fc(c5_relu->output);

	// classifire
				fc_out->forward_fc(c5_relu->output);

				estimate->forward_Softmax(fc_out->output);
				y_hat_vec_test[m] = estimate->y_hat; 	

				error += Cross_entropy(y_hat_vec_test[m], testset[m].out);	
		} // end testing set loop

	// display processing time and error
		std::cout << endl << t.elapsed() << "s elapsed." << std::endl;
		cout << "Test " << testing_size << ": this is epoch: " << epoch << " accuracy (%) = " << 100 - ((-1 / n) * error)  << endl;
		// restart display
		t.restart();
		disp1.restart(training_size);
	}
    printf("this is y hat [4]\n"); print_tensor(y_hat_vec_test[4]);
    printf("this is y correct [4]\n"); print_tensor(testset[4].out);
	printf("this is weight conv1\n"); print_tensor_vector(c1->W);
}

/*
// previous version (chage forward input definition)
int main()
{
	int pad =1;
	int stride =2;
	int Wsize = 6;
	int Nfilter = 2;

	float error = 0.0;

 	vector<case_t> dataset = read_train_cases();
 	vector<case_t> testset = read_test_cases();

    std::vector<tensor_t<float>> y_hat_vec, y_hat_vec_test;
    int size_x = dataset[0].out.size.x; int size_y = dataset[0].out.size.y ; int size_z = dataset[0].out.size.z;

    tensor_t<float> y_hat(size_x, size_y, size_z );
// set y hat = 0
	for (int i = 0; i < (int)dataset.size(); ++i)
	{
		y_hat = create_tensor(size_x, size_y, size_z, 0);
		y_hat_vec.push_back(y_hat);
	}

// set y hat test = 0
	tensor_t<float> y_hat_test(testset[0].out.size.x, testset[0].out.size.y, testset[0].out.size.z );
	for (int i = 0; i < (int)testset.size(); ++i)
	{
		y_hat_test = create_tensor(testset[0].out.size.x, testset[0].out.size.y, testset[0].out.size.z, 0);
		y_hat_vec_test.push_back(y_hat_test);
	}

// TEST LENET -5
// layer conv1 5x5 , 6 filters, stride 1, pad 0
	conv_layer * c1 = new conv_layer(5, 6, 1, 0, dataset[0].data.size); //Wsize, Nfilter, stride, pad, dataset[0].data.size
	relu_layer * c1_relu = new relu_layer(c1->output.size);

// pooling 1 2x2, stride 2 // max :0, average:1
	pool_layer * s2 = new pool_layer(2, 2, 1, c1_relu->output.size ); //(poolsize, poolstride, mode, conv1->output.size);

// conv2 5x5, 16 filters, stride 1, pad 0
	conv_layer * c3 = new conv_layer(5, 16, 1, 0, s2->output.size);
	relu_layer * c3_relu = new relu_layer(c3->output.size);

// pooling 2 2x2, stride 2 , mode 0 (max)
	pool_layer * s4 = new pool_layer(2, 2, 1, c3_relu->output.size);

// conv3 5x5, 120 filters, stride 1, pad 0 (now 4x4 bcoz : input change from 32x32 to 28x28)
	conv_layer * c5 = new conv_layer(4, 120, 1, 0, s4->output.size);
	relu_layer * c5_relu = new relu_layer(c5->output.size);

// FC layer
	fc_layer * fc6 = new fc_layer(84, c5_relu->output.size); // (Nclass, inputsize)
//classifier
	fc_layer * fc_out = new fc_layer(10, fc6->output.size);
	classifier * estimate = new classifier(fc_out->output.size);

//gradient Z
	tensor_t<float>  dE(1,1, fc6->output.size.z);
	tensor_t<float> dZ(1,1, fc6->output.size.z);


//start training
float n  = 0;
int testing_size = (int)testset.size(); // 100
int training_size = (int)dataset.size();//(int)dataset.size();// 2000
int mini_batch = 16;

// Thread add

// Display
 boost::progress_display disp1(training_size);
 boost::timer t;
	for (int epoch = 0; epoch < 10 ;++epoch)
	{			
		error = 0.0;
		n = 0;
		
		for (int k = 0; k < training_size ; k += mini_batch)  // for (int k = 0; k < training_size / mini_batch ; ++k) 
		{	
		 
		 for (int i = k ; i < mini_batch + k ; ++i) //for (int i = mini_batch * k; i < mini_batch * k + mini_batch ; ++i)
			{
				n++;
	// Conv c1
				c1->input = dataset[i].data; c1->Conv_forward(); 
				c1_relu->input = c1->output; c1_relu->Forward_ReLu();
	// Pool s2
				s2->input = c1_relu->output; s2->forward_pooling();
	// Conv c3
				c3->input = s2->output; c3->Conv_forward();
				c3_relu->input = c3->output; c3_relu->Forward_ReLu();
	// Pool s4
				s4->input = c3_relu->output; s4->forward_pooling();
	// Conv c5
				c5->input = s4->output; c5->Conv_forward();
				c5_relu->input = c5->output; c5_relu->Forward_ReLu();
	// FC 
				fc6->input = c5_relu->output; fc6->forward_fc();

	// classifire
				fc_out->input = fc6->output; fc_out->forward_fc();

				estimate->input = fc_out->output; estimate->Softmax();
				y_hat_vec[i] = estimate->y_hat; 	

				error += Cross_entropy(y_hat_vec[i], dataset[i].out);


	///////////////////  bacward /////////////////////////		

				estimate->Back_Softmax(dataset[i].out);
				dZ = estimate->gradient_dA;

				fc_out->backward_fc(dZ);

	// FC
				fc6->backward_fc(fc_out->gradient.dA);
	// conv c5
				c5_relu->Backward_ReLu(fc6->gradient.dA);
			    c5->Conv_backward(c5_relu->gradient_dA);  
	// Pool s4
			    s4->Backward_pooling(c5->gradient.dA);
	// Conv c3
			    c3_relu->Backward_ReLu(s4->gradient_dA);
			    c3->Conv_backward(c3_relu->gradient_dA);
	// Pool s2
			    s2->Backward_pooling(c3->gradient.dA);
	// Conv c1
			    c1_relu->Backward_ReLu(s2->gradient_dA);
			    c1->Conv_backward(c1_relu->gradient_dA);
		
		}	
		
	// update weights 
			    fc_out->fc_weight_update() ;// fc_out->fc_weight_update();
				fc6->fc_weight_update();
				c5->conv_weight_update();
				c3->conv_weight_update();
				c1->conv_weight_update();

	// display update	
 		 disp1 += mini_batch;	
		}
		// display processing time and error
		std::cout << endl << t.elapsed() << "s elapsed." << std::endl;
		cout << "Train " << training_size << ": this is epoch: " << epoch << " error (%) = " << ((-1 / n) * error)  << endl;
		// restart timer
		t.restart();

////////////////////////////////////////////////////////////
// Test model
////////////////////////////////////////////////////////////
		 error = 0.0; n = 0;
		for (int m = 0; m < testing_size; ++m)
		{
			n++;
			// Conv c1
				c1->input = testset[m].data; c1->Conv_forward(); 
				c1_relu->input = c1->output; c1_relu->Forward_ReLu();
	// Pool s2
				s2->input = c1_relu->output; s2->forward_pooling();
	// Conv c3
				c3->input = s2->output; c3->Conv_forward();
				c3_relu->input = c3->output; c3_relu->Forward_ReLu();
	// Pool s4
				s4->input = c3_relu->output; s4->forward_pooling();
	// Conv c5
				c5->input = s4->output; c5->Conv_forward();
				c5_relu->input = c5->output; c5_relu->Forward_ReLu();
	// FC 
				fc6->input = c5_relu->output; fc6->forward_fc();

	// classifire
				fc_out->input = fc6->output; fc_out->forward_fc();

				estimate->input = fc_out->output; estimate->Softmax();
				y_hat_vec_test[m] = estimate->y_hat; 	

				error += Cross_entropy(y_hat_vec_test[m], testset[m].out);	
		} // end testing set loop

	// display processing time and error
		std::cout << endl << t.elapsed() << "s elapsed." << std::endl;
		cout << "Test " << testing_size << ": this is epoch: " << epoch << " accuracy (%) = " << 100 - ((-1 / n) * error)  << endl;
		// restart display
		t.restart();
		disp1.restart(training_size);
	}
    printf("this is y hat [4]\n"); print_tensor(y_hat_vec_test[4]);
    printf("this is y correct [4]\n"); print_tensor(testset[4].out);
	printf("this is weight conv1\n"); print_tensor_vector(c1->W);
}
*/
