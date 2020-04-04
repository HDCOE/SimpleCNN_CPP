

network* create_Cifar()
{
	int workspace_size = 0;
	//network *net = load_network(2, "yolov2.weights", 0);
	network *net = make_network(10);

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
	layer c1 = parse_convolutional(3, 32, 1, 1, params, LEAKY, 1);
	layer p1 = parse_maxpool(2,2,0,c1.params);

	layer c2 = parse_convolutional(1,16,1,0,p1.params, LEAKY, 1);

	layer c3 = parse_convolutional(3,64,1,1, c2.params, LEAKY, 1);
	layer p3 = parse_maxpool(2,2,0,c3.params);

	layer c4 = parse_convolutional(1,32,1,0, p3.params, LEAKY, 1);// pad

	layer c5 = parse_convolutional(3,128,1,1, c4.params, LEAKY, 1);

	layer c6 = parse_convolutional(1,64,1,0, c5.params, LEAKY, 1);

	layer c7 = parse_convolutional(1,10,1,0, c6.params, LEAKY, 1);


	layer p8 = parse_avgpool(7,1,0,c7.params);
	
	net->layers[0] = c1;
	net->layers[1] = p1;
	net->layers[2] = c2;
	net->layers[3] = c3;
	net->layers[4] = p3;
	net->layers[5] = c4;
	net->layers[6] = c5;
	net->layers[7] = c6;
	net->layers[8] = c7;
	net->layers[9] = p8;


// choose the largest space from every layer
	workspace_size = c5.workspace_size;
	net->workspace = (data_t *)calloc(1, workspace_size);

    net->outputs = p8.outputs;
    net->output = p8.output;

    load_network(net,"/home/hadee/Work/darknet/darknet/Dataset/cifar10/cifar/cifar_small.weights" );
	save_weights(net,"/home/hadee/Work/darknet/darknet/Dataset/cifar10/cifar/cifar_fix.weights");

return net;
}
