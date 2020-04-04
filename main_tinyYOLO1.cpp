/* 

g++ main_tinydarknet.cpp -o main -lm -Wno-write-strings; ./main; rm main

g++ -I/home/hadee/Work/SimpleCNN_CPP/HLS_lib/  main_tinyYOLO1.cpp  -o main -lm -Wno-write-strings; ./main; rm main
*/

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

#define mini_batch 16

#include "HLS_lib/ap_fixed.h"

//#include "YOLOsrc/half.hpp"

#include "Tensor.h"
// add yolo
#include "YOLOsrc/darknet_tiny.h"
#include "tiny_darknet.h"
//#include "Graph/draw_graph.h"


#include "lenet.h"
#include "cifar10.h"
using namespace std;

image load_dataset(char * filename)
{
	///////////////////////////////////////////////////////////////////
//////////////////// load img resize img begin  ////////////////////
	char buff[256]; 
    char *input_imgfn = buff;
	strncpy(input_imgfn, filename, 256);

	printf("Input img:%s\n",input_imgfn);
	//image im = load_image_stb(input_imgfn, 3);//3 channel img

	image im = load_image_color(input_imgfn, 0, 0);

	printf("img w=%d,h=%d,c=%d\n",im.w,im.h,im.c);
	//image sized = letterbox_image(im, 448, 448);  //darknet19 input is 256x256
	image sized = resize_image(im, 448,448);
/*	for (int i = 0; i < 50; ++i)
	{
		printf("data[%d] %f\n",i, (float)sized.data[i]);
	}
*/	
	save_image_png(sized, "sized");

/////////////////// load img resize img end   //////////////////////
	return sized;
}


int simple_one_layer();
int main_t();
char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

#define size_in 3

void write_2file(ofstream * file, data_t * value, int size, char * name )
{
	
	printf(" %s number of variable: %d\n", name, size );
	for (int i = 0; i < size; ++i) //l.nweights
	{
		ap_fixed<16,5> x;
		x = value[i];
		file->write((char*)&x,sizeof(float));
		
		if (i < 20)
			printf("%s [%d]: %f\n",name, i, (float)x );
	}
}
void write_weight(ofstream *outfile, layer l)
{
	write_2file(outfile, l.weights, l.nweights, "weight");
	write_2file(outfile, l.biases, l.n, "bias");
	
	if (l.batch_normalize)
	{
		write_2file(outfile, l.scales, l.n, "scale");
		write_2file(outfile, l.rolling_mean, l.n, "mean");
		write_2file(outfile, l.rolling_variance, l.n, "variance");
	}
}

void write_weight_FC(ofstream *outfile, layer l)
{
	write_2file(outfile, l.weights, l.outputs*l.inputs, "weight");
	write_2file(outfile, l.biases, l.outputs, "bias");
	
	if (l.batch_normalize)
	{
		write_2file(outfile, l.scales, l.outputs, "scale");
		write_2file(outfile, l.rolling_mean, l.outputs, "mean");
		write_2file(outfile, l.rolling_variance, l.outputs, "variance");
	}

}

void Dataset2file(image im)
{
	if( remove( "image.dat" ) != 0 )
		perror( "Error deleting file" );
	else
		puts( "File successfully deleted" );

	ofstream outfile ("image.dat",std::ofstream::binary);

	int size = im.w * im.h * im.c;

	data_t x;

printf("size %d\n",size );
	for (int i = 0; i < size; ++i)
	{
		x = im.data[i];
		outfile.write((char*)&x,sizeof(float));
	}

	outfile.close();

}

int main()
{


 network *net = create_tinyDarknet();

 	//network *net = create_Lenet();

	//network *net = create_Cifar();

  //layer l = net->layers[0];
  //write_weight(l);
/*
	if( remove( "modelfix.dat" ) != 0 )
		perror( "Error deleting file" );
	else
		puts( "File successfully deleted" );

	ofstream outfile ("modelfix.dat",std::ofstream::binary);

 for(int i = 0; i < net->n; ++i)
 {
    net->index = i;
    layer l = net->layers[i]; 
    printf("layer [%d]\n",i);

    if (l.type == CONVOLUTIONAL)
    	write_weight(&outfile, l);
    else if(l.type == CONNECTED)
    	write_weight_FC(&outfile, l);
 }
 
	outfile.close();
*/

/*
image dataset1 = load_dataset("dog.jpg");

Dataset2file(dataset1);
*/


	//main_t();
}

int simple_one_layer()
{
	
	network *net = make_network(1);
	 tensor_t<data_t> X(size_in,size_in,1);


	net->h = size_in; net->w = size_in; net->c = 1;
	net->inputs =net->h * net->w * net->c;
	net->batch = 1;
	net->time_steps = 1;

    net->input = (data_t *)calloc(net->inputs*net->batch, sizeof(data_t));

	net->gpu_index = -1;

	size_params params;

  	params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;

    layer fc1 = parse_convolutional(2, 1, 1,1 , params, LEAKY, 0);  // parse_connected(3, LINEAR, params ); //parse_convolutional(3, 16, 1, 1, params, LEAKY, 0);

    net->layers[0] = fc1;

    net->outputs = fc1.outputs;
    net->output = fc1.output;


    int workspace = fc1.workspace_size;
    net->workspace = (data_t *)calloc(1, workspace);

	data_t *predictions = network_predict(net, X.data);

	//print_tensor(X);

	for (int i = 0; i < 10; ++i)
	{
		printf("weights %d %f\n", i, (float)fc1.weights[i]);
	}
	for (int i = 0; i < 4; ++i)
	{
		printf("output %d : %f\n",i, (float)net->layers[net->n -1].output[i] );
	}
return 0;

}

int main_t()
{

	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(16);

	image sized = load_dataset("dog.jpg");
	image im = load_image_color("dog.jpg", 0, 0);

	net->inputs = 448*448*3;
	net->h = 448; net->w = 448; net->c = 3;
	net->batch = 1;
	net->time_steps = 1;

    net->input = (data_t *)calloc(net->inputs*net->batch, sizeof(data_t));

	net->gpu_index = -1;

	size_params params;

  	params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    //params.net = net;

//size, filter, stride, pad, param, activate, batch
	layer c1 = parse_convolutional(3, 16, 1, 1, params, LEAKY, 1); 
	layer p1 = parse_maxpool(2,2,0,c1.params);

	layer c2 = parse_convolutional(3,32,1,1,p1.params, LEAKY, 1);
	layer p2 = parse_maxpool(2,2,0,c2.params);

	layer c3 = parse_convolutional(3,64,1,1, p2.params, LEAKY, 1);
	layer p3 = parse_maxpool(2,2,0,c3.params);

	layer c4 = parse_convolutional(3,128,1,1, p3.params, LEAKY, 1);// pad
	layer p4 = parse_maxpool(2,2,0,c4.params);

	layer c5 = parse_convolutional(3,256,1,1, p4.params, LEAKY, 1);
	layer p5 = parse_maxpool(2,2,0,c5.params);

	layer c6 = parse_convolutional(3,512,1,1, p5.params, LEAKY, 1);
	layer p6 = parse_maxpool(2,2,0,c6.params);

	layer c7 = parse_convolutional(3,1024,1,1, p6.params, LEAKY, 1);
	layer c8 = parse_convolutional(3,256,1,1, c7.params, LEAKY, 1);


//(int outputs, ACTIVATION activate , size_params params)
 	layer fc1 = parse_connected(1470, LINEAR, c8.params );
	//int coords, int classes, int side, float jitter, size_params params
	layer d1 = parse_detection(4, 20, 7, fc1.params);

	net->layers[0] = c1;
	net->layers[1] = p1;
	net->layers[2] = c2;
	net->layers[3] = p2;
	net->layers[4] = c3;
	net->layers[5] = p3;
	net->layers[6] = c4;
	net->layers[7] = p4;
	net->layers[8] = c5;
	net->layers[9] = p5;
	net->layers[10] = c6;
	net->layers[11] = p6;
	net->layers[12] = c7;
	net->layers[13] = c8;
	net->layers[14] = fc1;
	net->layers[15] = d1;

// choose the largest space from every layer
	workspace_size = c2.workspace_size;
	net->workspace = (data_t *)calloc(1, workspace_size); 

    net->outputs = d1.outputs;
    net->output = d1.output;


	load_network(net,"/home/hadee/Work/darknet/darknet/tiny-yolov1.weights" );

	data_t *predictions = network_predict(net, sized.data);


   for (int i = 0; i < 20; ++i)
	    {
	      //  printf("weight layer conv0 %d : %f\n",i, (float)net->layers[0].weights[i] );
	    }

	  for (int i = 0; i < 20; ++i)
	    {
	        printf("layer fc %d : %f\n",i, (float)net->layers[net->n -2].output[i] );
	    }

	int nboxes = 0;
	float thresh = 0.1; //.2
	 printf("start get network box\n");
    detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);


    float nms = 0.40;
    printf("finish get network box\n");
    layer l = net->layers[net->n-1];
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    
         for (int i = 0; i < nboxes; ++i)
        {
           // printf("dets[%d] : x %f, y%f, w%f, h%f \n", i, (float)dets[i].bbox.x, (float) dets[i].bbox.y,  (float)dets[i].bbox.w, (float)dets[i].bbox.h);
        }

    //image **alphabet = load_alphabet();
      image **alphabet;

       draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
      //  save_image(sized, "predictions");
 		
 		save_image_png(im, "predictions");
        free_detections(dets, nboxes);
        free_image(sized);
       

}
/*
int workspace_size = 0;
	network *net = make_network(2);
	net->inputs = 6*6*1;
	net->h = 6; net->w = 6; net->c = 1;
	net->batch = 1;
	net->time_steps = 1;

    net->input = (float *)calloc(net->inputs*net->batch, sizeof(float));

	net->gpu_index = -1;

	size_params params;

  	params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    //params.net = net;



    layer l;
	memset(&l,0,sizeof(layer));
	l = parse_convolutional(3, 1, 0, 1 , 1, params);  //int w_size, int filters, int pad, int stride , int activation,  size_params params)  

	params.h = l.out_h;
    params.w = l.out_w;
    params.c = l.out_c;
    params.inputs = l.outputs;

	layer l2;
	memset(&l2,0,sizeof(layer));
	l2 = parse_avgpool(2,2,0, l.params); // (size, stride, padding)

	net->layers[0] = l;
	net->layers[1] = l2;




// choose the largest space from every layer
	workspace_size = l.workspace_size;
	net->workspace = (float *)calloc(1, workspace_size); 

    net->outputs = l2.outputs;
    net->output = l2.output;

	tensor_t<float> data_in( 6, 6, 1 ) ;
	tensor_t<float> convW(3,3,1);
	tensor_t<float> out_conv(4,4,1);

	tensor_t<float> data_out(2,2,1);

	convW.data = l.weights;

	data_in(0,0,0) = 3;
	print_tensor(convW);
	print_tensor(data_in);

	float *predictions = network_predict(net, data_in.data);

	out_conv.data = net->layers[0].output;

	print_tensor(out_conv);

	data_out.data = predictions;

	print_tensor(data_out);*/