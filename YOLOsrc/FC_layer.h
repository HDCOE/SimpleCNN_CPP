

void forward_connected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, (data_t)0, l.output, (data_t)1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    data_t *a = net.input;
    data_t *b = l.weights;
    data_t *c = l.output;

    gemm(0,1,m,n,k,(data_t)1,a,k,b,k,(data_t)1,c,n);

    // 0,1,batch,outputs,inputs, 1, input, inputs, weight, inputs,1, output, outputs

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}


layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l;
    memset(&l,0,sizeof(layer));
  
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = (data_t*)calloc(batch*outputs, sizeof(data_t));

    l.weights = (data_t*)calloc(outputs*inputs, sizeof(data_t));
    l.biases = (data_t*)calloc(outputs, sizeof(data_t));

    l.forward = forward_connected_layer;
   
    //float scale = 1./sqrt(inputs);
  // float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = 1;//scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

 /*   if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }
*/
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);

    /// return param
    l.params.h = l.out_h;
    l.params.w = l.out_w;
    l.params.c = l.out_c;
    l.params.inputs = l.outputs;
    l.params.batch = batch;
    
    return l;
}

layer parse_connected(int outputs, ACTIVATION activate , size_params params)
{

    int batch_normalize = 0;

   // params.net->adam = 0;

    layer l = make_connected_layer(params.batch, params.inputs, outputs, activate, batch_normalize, 0);

    return l;
}