
network* mnist_BNN()
{
	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(7);

	//image sized = load_dataset("000069.jpg");
	//image im = load_image_color("000069.jpg", 0, 0);
	int in_size = 28;
	net->inputs = in_size*in_size*3;
	net->h = in_size; net->w = in_size; net->c = 3;
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

    //size, filter, stride, pad, param, activate, batch
	layer c1 = parse_convolutional(5, 32, 1, 1, params, LEAKY, 0); 
	layer p1 = parse_maxpool(2,2,0,c1.params);

	layer c2 = parse_convolutional(5,64,1,1,p1.params, LEAKY, 0);
	layer p2 = parse_maxpool(2,2,0,c2.params);

	layer fc1 = parse_connected(1470, LINEAR, p2.params );

	net->layers[0] = c1;
	net->layers[1] = p1;
	net->layers[2] = c2;
	net->layers[3] = p2;
	net->layers[4] = fc1;

	workspace_size = c2.workspace_size;
	net->workspace = (data_t *)calloc(1, workspace_size); 

    net->outputs = fc1.outputs;
    net->output = fc1.output;

    load_network(net,"/home/hadee/Work/darknet/darknet/Dataset/mnist/mnist_lenetBNN.weights" );

	return net;

}
