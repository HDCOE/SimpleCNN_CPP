




int convolutional_out_width(layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}
int convolutional_out_height(layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

static size_t get_workspace_size(layer l){

    printf(" workspace size %ld  ", l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(data_t));
 
 return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(data_t);
    //return (size_t)l.out_h*l.out_w*l.size*l.size*l.c;
}


void add_bias(data_t *output, data_t *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void normalize_cpu(data_t *x, data_t *mean, data_t *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = ((float)x[index] - (float)mean[f])/(sqrt((float)variance[f]) + .000001f);
            }
        }
    }
}
void scale_bias(data_t *output, data_t *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}
void forward_batchnorm_layer(layer l, network net)//for conv
{
    normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);   
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}


void forward_convolutional_layer(layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, (data_t)0, l.output, (data_t)1);

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            data_t *a = l.weights + j*l.nweights/l.groups; // weight
            data_t *b = net.workspace; //size of all weights
            data_t *c = l.output + (i*l.groups + j)*n*m; // output
            data_t *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;  // input

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b); 
            }
            gemm(0,0,m,n,k,(data_t)1,a,k,b,n,(data_t)1,c,n);
        }
    }


  if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } 
  else 
    {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
     //add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
     activate_array(l.output, l.outputs*l.batch, l.activation);
     
}

layer make_convolutional_layer(int batch, int w_size, int n , int stride, int padding, int h, int w, int c, ACTIVATION activate, int batch_normalize)
{
    int i;
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;//number filter

    l.binary = 0;
    l.batch = batch;
    l.stride = stride;
    l.size = w_size;
    l.pad = padding ;
    l.groups = 1;
    int groups = 1;
    l.batch_normalize = batch_normalize;


    l.weights = (data_t *)calloc(c/groups*n*w_size*w_size, sizeof(data_t));
    l.biases = (data_t *)calloc(n, sizeof(data_t));


    l.nweights = c/groups*n*w_size*w_size;
    l.nbiases = n;

    if(l.batch_normalize){
        l.scales = (data_t *)calloc(n, sizeof(data_t));
        l.rolling_mean = (data_t *)calloc(n, sizeof(data_t));
        l.rolling_variance = (data_t *)calloc(n, sizeof(data_t));
    }

 for(i = 0; i < l.nweights; ++i) l.weights[i] = 1; //random

 for(i = 0; i < l.nbiases; ++i) l.biases[i] = 0;
    
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (data_t *)calloc(l.batch * l.outputs, sizeof(data_t));
    l.forward = forward_convolutional_layer;  //point to forward function


    l.workspace_size = get_workspace_size(l);

    // ACTIVATION relu = ac;
    l.activation = activate;
   // l.dontload = 0;

    printf(" nweights %d nbiases %d\n", l.nweights, l.nbiases );
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, w_size, w_size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c * l.out_h*l.out_w)/1000000000.);

/// return param
    l.params.h = l.out_h;
    l.params.w = l.out_w;
    l.params.c = l.out_c;
    l.params.inputs = l.outputs;
    l.params.batch = batch;
    
    return l;
}

layer parse_convolutional(int w_size, int filters, int stride, int pad ,  size_params params, ACTIVATION activate ,int batch_normalize)
{

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;

    layer l = make_convolutional_layer(batch, w_size, filters , stride, pad,  h, w,  c, activate, batch_normalize);

    return l;
}