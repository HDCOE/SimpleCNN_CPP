

void transpose_matrix(data_t *a, int rows, int cols)
{
    data_t *transpose = (data_t*)calloc(rows*cols, sizeof(data_t));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(data_t));
    free(transpose);
}


void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
// add
    for (int i = 0; i < 20; ++i)
    {
        printf("from file weight[%d] : %f\n",i, (float)l.weights[i] );
    }
// end    
}
void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        load_convolutional_weights_binary(l, fp);
        return;
    }
    // if(l.numload) l.n = l.numload;
    float  *weights, *biases, *scales, *rolling_mean, *rolling_variance; //create float variable 
    int num = l.c/l.groups*l.n*l.size*l.size;

    weights = (float *)calloc(num, sizeof(float)); // create weight
    biases = (float *)calloc(l.n,sizeof(float)); // create bias

    //fread(l.biases, sizeof(float), l.n, fp);
//copy bias and move to 16
    fread(biases, sizeof(float), l.n, fp);
    copy_tensor_32(l.n,biases,l.biases);

    if (l.batch_normalize){

        scales = (float *)calloc(l.n, sizeof(float));
        rolling_mean = (float *)calloc(l.n, sizeof(float));
        rolling_variance = (float *)calloc(l.n, sizeof(float));

        //fread(l.scales, sizeof(float), l.n, fp);
        //fread(l.rolling_mean, sizeof(float), l.n, fp);
        //fread(l.rolling_variance, sizeof(float), l.n, fp);
        fread(scales, sizeof(float), l.n, fp); copy_tensor_32(l.n,scales,l.scales);
        fread(rolling_mean, sizeof(float), l.n, fp); copy_tensor_32(l.n,rolling_mean,l.rolling_mean);
        fread(rolling_variance, sizeof(float), l.n, fp); copy_tensor_32(l.n,rolling_variance,l.rolling_variance);


        if(0){ //0
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ",(float) l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", (float)l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, (data_t)0, l.rolling_mean, (data_t)1);
            fill_cpu(l.n, (data_t)0, l.rolling_variance, (data_t)1);
        }
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ",(float) l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", (float)l.rolling_variance[i]);
            }
            printf("\n");
        }

        free(scales); free(rolling_mean); free(rolling_variance);
    }
      //fread(l.weights, sizeof(float), num, fp);
    // read from float tohen move to half
    fread(weights, sizeof(float), num, fp);
    copy_tensor_32(num,weights,l.weights);
    
    /*
    for (int i = 0; i < 20; ++i)
    {
       printf("weight[%d] %g \n",i, (float)l.weights[i]);
    }
        printf("end layer\n");

    */


    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
    free(weights);
    free(biases);
}

data_t sum_array(data_t *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += (float)a[i];
    return (data_t)sum;
}

data_t mean_array(data_t *a, int n)
{
    return (data_t)(sum_array(a,n)/n);
}
data_t variance_array(data_t *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += ((float)a[i] - mean)*((float)a[i]-mean);
    float variance = sum/n;
    return (data_t)variance;
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{

    float  *weights, *biases, *scales, *rolling_mean, *rolling_variance; //create float variable 
    int num = l.outputs*l.inputs;

   
    //fread(l.biases, sizeof(float), l.outputs, fp);
    //fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
// add
    biases = (float *)calloc(l.outputs,sizeof(float)); // create bias
    weights = (float *)calloc(num, sizeof(float)); // create weight
    fread(biases, sizeof(float), l.outputs, fp);  copy_tensor_32(l.outputs,biases,l.biases);
    fread(weights, sizeof(float), num, fp); copy_tensor_32(num,weights,l.weights);
// end add

  /*  for (int i = 0; i < 20; ++i)
    {
       printf("fc layer weights[%d] %f bias %f\n",i, (float)l.weights[i], (float)l.biases[i] );
    }
    */
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    printf("Biases: %f mean %f variance\n", (float)mean_array(l.biases, l.outputs), (float)variance_array(l.biases, l.outputs));
    printf("Weights: %f mean %f variance\n", (float)mean_array(l.weights, l.outputs*l.inputs), (float)variance_array(l.weights, l.outputs*l.inputs));
    
    if (l.batch_normalize)
    {
        //fread(l.scales, sizeof(float), l.outputs, fp);
        //fread(l.rolling_mean, sizeof(float), l.outputs, fp);
       // fread(l.rolling_variance, sizeof(float), l.outputs, fp);
    // add
        scales = (float *)calloc(l.outputs, sizeof(float));
        rolling_mean = (float *)calloc(l.outputs, sizeof(float));
        rolling_variance = (float *)calloc(l.outputs, sizeof(float));

        fread(scales, sizeof(float), l.outputs, fp); copy_tensor_32(l.outputs,scales,l.scales);
        fread(rolling_mean, sizeof(float), l.outputs, fp); copy_tensor_32(l.outputs,rolling_mean,l.rolling_mean);
        fread(rolling_variance, sizeof(float), l.outputs, fp); copy_tensor_32(l.outputs,rolling_variance,l.rolling_variance);
    // end add

        free(scales); free(rolling_mean); free(rolling_variance);

        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }

    free(weights); free(biases);
}


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

    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } 
    else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }

    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        //if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }

        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_network(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}


//////////////////// save weight ////////////////////////////

void floatfixt(float * out, int size)
{
    fix16 fvalue;
    float err = 0;

    for(int i = 0;i < size; i++)
    {
        float previous = out[i];
        fvalue = out[i];
        out[i] = fvalue;

        err += abs(out[i] - previous); 

    }

    printf("error %f \n", err / size);
}

void save_convolutional_weights(layer l, FILE *fp)
{
    int num = l.nweights;

    floatfixt(l.biases, l.n);
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){

       // floatfixt(l.scales, l.n);
       // floatfixt(l.rolling_mean , l.n);
      //  floatfixt(l.rolling_variance, l.n);

        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }

    floatfixt(l.weights, num);
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{

    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{

    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        }
    }

    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}