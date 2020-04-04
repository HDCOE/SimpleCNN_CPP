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
#include "Tensor.h"
// add yolo
#include "YOLOsrc/darknet_tiny.h"
//#include "darknet19.h"

//#include "Graph/draw_graph.h"
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
	image sized = letterbox_image(im, 256, 256);  //darknet19 input is 256x256
	save_image_png(sized, "sized");

/////////////////// load img resize img end   //////////////////////
	return sized;
}
int main()
{
	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(25);

	image sized = load_dataset("eagle.jpg");

	net->inputs = 256*256*3;
	net->h = 256; net->w = 256; net->c = 3;
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


	//layer l = parse_convolutional(3, 1, 0, 1 , 1, params);  //int w_size, int filters, stride, int pad , int activation,  size_params params)  
	//layer l2 = parse_avgpool(2,2,0, l.params); // (size, stride, padding)


	layer c1 = parse_convolutional(3, 32, 1, 1, params, LEAKY, 1);
	layer p1 = parse_maxpool(2,2,0,c1.params);
	layer c2 = parse_convolutional(3,64,1,1,p1.params, LEAKY, 1);
	layer p2 = parse_maxpool(2,2,0,c2.params);

	layer c3 = parse_convolutional(3,128,1,1, p2.params, LEAKY, 1);
	layer c4 = parse_convolutional(1,64,1,0, c3.params, LEAKY, 1);// pad
	layer c5 = parse_convolutional(3,128,1,1, c4.params, LEAKY, 1);
	layer p3 = parse_maxpool(2,2,0,c5.params);

	layer c6 = parse_convolutional(3,256,1,1, p3.params, LEAKY,1);
	layer c7 = parse_convolutional(1,128,1,0, c6.params, LEAKY,1);//
	layer c8 = parse_convolutional(3,256,1,1, c7.params, LEAKY,1);
	layer p4 = parse_maxpool(2,2,0, c8.params);

	layer c9 = parse_convolutional(3,512,1,1, p4.params, LEAKY,1);
	layer c10 = parse_convolutional(1,256,1,0, c9.params, LEAKY,1); //
	layer c11 = parse_convolutional(3,512,1,1, c10.params, LEAKY,1);
	layer c12 = parse_convolutional(1,256,1,0, c11.params, LEAKY,1);
	layer c13 = parse_convolutional(3,512,1,1, c12.params, LEAKY,1);
	layer p5 = parse_maxpool(2,2,0, c13.params);

	layer c14 = parse_convolutional(3,1024,1,1, p5.params, LEAKY,1);
	layer c15 = parse_convolutional(1,512,1,0, c14.params, LEAKY,1); //
	layer c16 = parse_convolutional(3,1024,1,1,c15.params, LEAKY,1);
	layer c17 = parse_convolutional(1,512,1,0, c16.params, LEAKY,1); //
	layer c18 = parse_convolutional(3,1024,1,1,c17.params, LEAKY,1);
	layer c19 = parse_convolutional(1,1000,1,0, c18.params, LINEAR,0); //
	layer p6 = parse_avgpool(8,1,0, c19.params);

	net->layers[0] = c1;
	net->layers[1] = p1;
	net->layers[2] = c2;
	net->layers[3] = p2;
	net->layers[4] = c3;
	net->layers[5] = c4;
	net->layers[6] = c5;
	net->layers[7] = p3;
	net->layers[8] = c6;
	net->layers[9] = c7;
	net->layers[10] = c8;
	net->layers[11] = p4;
	net->layers[12] = c9;
	net->layers[13] = c10;
	net->layers[14] = c11;
	net->layers[15] = c12;
	net->layers[16] = c13;
	net->layers[17] = p5;
	net->layers[18] = c14;
	net->layers[19] = c15;
	net->layers[20] = c16;
	net->layers[21] = c17;
	net->layers[22] = c18;
	net->layers[23] = c19;
	net->layers[24] = p6;





// choose the largest space from every layer
	workspace_size = c2.workspace_size;
	net->workspace = (float *)calloc(1, workspace_size); 

    net->outputs = p6.outputs;
    net->output = p6.output;


	 load_weights(net,"/home/hadee/Work/darknet_bin/weights.bin","/home/hadee/Work/darknet_bin/bias.bin");
	
	//load_network(net,"bin/darknet19.weights" );


	tensor_t<float> data_in( 256, 256, 3 ) ;
	tensor_t<float> convW(3,3,32); 
	tensor_t<float> bias(1,1,1);
	tensor_t<float> out_conv(4,4,1);

	tensor_t<float> data_out(1,1,1000);

	data_in.data = sized.data;

        for (int i = 0; i < 32; ++i)
        {
            printf("conv variance [%d] : %g\n", i, net->layers[23].weights[i]);
        }
	
	//convW.data = net->layers[0].weights;
	//print_tensor(convW);
	//print_tensor(bias);
	//print_tensor(data_in);

	float *predictions = network_predict(net, sized.data);

	out_conv.data = net->layers[0].output;

	//print_tensor(out_conv);


   // data_out.data = predictions;
	softmax(predictions, net->outputs, data_out.data);
	

  // print_tensor(data_out);

      for (int i = 0; i < 1002; ++i)
        {
            printf(" output [%d] %f :softmax out [%d] : %f \n",i, net->layers[24].output[i],i, data_out.data[i] );
        }
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