
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    data_t *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
//    tree *hierarchy;

    data_t *input;
    data_t *truth;
    data_t *delta;
    data_t *workspace;
    int train;
    int index;
    data_t *cost;
    data_t clip;
} network;
/*
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

*/





network *make_network(int n)
{
    network *net = (network *)calloc(1, sizeof(network));
    net->n = n;
    net->layers = (layer *)calloc(net->n, sizeof(layer));
    net->seen = (size_t *)calloc(1, sizeof(size_t));
    net->t    = (int *)calloc(1, sizeof(int));
    net->cost = (data_t *)calloc(1, sizeof(data_t));
    return net;
}


void forward_network(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        l.forward(l, net);
        net.input = l.output;

      printf("layer [%d]\n",i);
    }
 
}

data_t *network_predict(network *net, data_t *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    data_t *out = net->output;
    *net = orig;
    return out;
}

/*
void load_convolutional_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights, sizeof(float), num, fp);

}


///// load network
void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    printf("major=%d;minor=%d;revision=%d\n",major,minor,revision);// 0 2 0
    printf("if true ro false:%d\n",(major*10 + minor) >= 2 && major < 1000 && minor < 1000);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        //fread(net->seen, sizeof(size_t), 1, fp);
        fread(net->seen, sizeof(size_t), 1, fp);
        fread(net->seen, sizeof(size_t), 1, fp);
    }else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }

    //printf("sizeof(size_t)=%u\n",sizeof(size_t));// in my PC is 4

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}



void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}
*/

float *extract_weight( char *filename)
{
    FILE *fp_w = fopen(filename, "rb");

    // obtain file size
    fseek (fp_w , 0 , SEEK_END);
    long lSize = ftell (fp_w);
    long Bsize = 0;
    rewind (fp_w);

    printf("\nfile: %s size:%ld \n", filename, lSize );

    // create buffer
    float *buffer = (float *)calloc(lSize /4 ,sizeof(float));

    fread(buffer, sizeof(float), lSize /4 , fp_w);
    fclose(fp_w);

    return buffer;
}

void load_weights(network *net, char *filename, char *bias_file)
{
    int weight_offset[26] = {864, 0, 18432, 0, 73728, 8192, 73728, 0, 294912, 32768, 294912,

                             0, 1179648, 131072, 1179648, 131072, 1179648, 0, 4718592, 524288, 

                             4718592, 524288, 4718592, 1024000, 0, 0 }; // number of weights in each layer,Ex first layer has 864 weights

    int bias_offset[26]={ 32, 0, 64, 0, 128, 64, 128, 0, 256, 128, 256, 0, 512, 256, 512, 256, 

                          512, 0, 1024, 512, 1024, 512, 1024, 1000,0,0 };

    int woffset = 0, boffset = 0;   

    float *Weight_buf  = extract_weight(filename);
    float *Bias_buf = extract_weight(bias_file);

    float *mean_buf = extract_weight("/home/hadee/Work/darknet_bin/means.bin");
    float *variance_buf = extract_weight("/home/hadee/Work/darknet_bin/variance.bin");
    float *scale_buf = extract_weight("/home/hadee/Work/darknet_bin/scales.bin");
                     
    for (int i = 0; i < net->n; ++i)
    {
        if (net->layers[i].type == CONVOLUTIONAL)
        {
            float * w  = Weight_buf+woffset;
            //weight load
            for (int idx = 0; idx < weight_offset[i]; ++idx)
            {
               net->layers[i].weights[idx] = w[idx];
            }
           // bias
            float * b = Bias_buf+boffset;
            float * m = mean_buf+boffset;
            float * v = variance_buf+boffset;
            float * sc = scale_buf+boffset;

            for (int idx = 0; idx < bias_offset[i]; ++idx)
            {
               net->layers[i].biases[idx] = b[idx];

               if (net->layers[i].batch_normalize)
               {
            // load mean and variance
                    net->layers[i].rolling_mean[idx] = m[idx];
                    net->layers[i].rolling_variance[idx] = v[idx];
                    net->layers[i].scales[idx] = sc[idx];
               }
            }
        }
        woffset += weight_offset[i];
         boffset += bias_offset[i];
    }
}

/*
network *load_network(int n, char *filename, int clear)
{
   // network *net = make_network(n);
   // load_weights(net, filename);
    

    //if(weights && weights[0] != 0){
    //    load_weights(net, weights);
    //}
    //if(clear) (*net->seen) = 0;
    //return net;
}
*/
