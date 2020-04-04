void softmax(data_t *input, int n, float temp, int stride, data_t *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if((float)input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp((float)input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] =  (float)output[i*stride]/(float)sum;//output[i*stride] /= (float)sum;
    }
}

void softmax_2(float *input, int n, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;

    //find max
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }

    //
    for(i = 0; i < n; ++i){
        float e = exp(input[i] - largest);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }

// find max id

    int max_id = 0;
    float max_out = output[0];

    for (int id = 0; id < n; ++id)
    {
        if(output[id]>max_out)
         {
            max_out = output[id];
            max_id = id;
         }
    }

    printf(" class id %d has value %f\n", max_id, max_out );
    //return max_id;
}
