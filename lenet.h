

network* create_Lenet()
{
	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(7);

	//image sized = load_dataset("000069.jpg");
	//image im = load_image_color("000069.jpg", 0, 0);

	net->inputs = 28*28*3;
	net->h = 28; net->w = 28; net->c = 3;
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
	layer c1 = parse_convolutional(5, 6, 1, 2, params, RELU, 0);
	layer p1 = parse_maxpool(2,2,0,c1.params);

	layer c2 = parse_convolutional(5,16,1,2,p1.params, RELU, 0);
	layer p2 = parse_maxpool(2,2,0,c2.params);

	layer c3 = parse_convolutional(5,120,1,2, p2.params, RELU, 0);

//(int outputs, ACTIVATION activate , size_params params)
 	layer fc1 = parse_connected(84, LINEAR, c3.params );
	layer fc2 = parse_connected(10, LINEAR, fc1.params );

	net->layers[0] = c1;
	net->layers[1] = p1;
	net->layers[2] = c2;
	net->layers[3] = p2;
	net->layers[4] = c3;
	
	net->layers[5] = fc1;
	net->layers[6] = fc2;

// choose the largest space from every layer
	workspace_size = c3.workspace_size;
	net->workspace = (data_t *)calloc(1, workspace_size);

    net->outputs = fc2.outputs;
    net->output = fc2.output;

    load_network(net,"/home/hadee/Work/darknet/darknet/Dataset/mnist/mnist_lenet3conv.weights" );

	save_weights(net,"/home/hadee/Work/darknet/darknet/Dataset/mnist/lenet_fix.weights");

return net;
}
