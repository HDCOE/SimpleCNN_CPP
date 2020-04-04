	
network* create_tinyDarknet()
{
	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(16);

	//image sized = load_dataset("000069.jpg");
	//image im = load_image_color("000069.jpg", 0, 0);

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


	load_network(net,"/home/hadee/Work/darknet/darknet/Dataset/VOCdevkit/tiny-yolov1.weights" );

	save_weights(net,"/home/hadee/Work/darknet/darknet/Dataset/VOCdevkit/yolo.weights");
return net;
}