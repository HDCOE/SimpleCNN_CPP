
void forward_avgpool_layer(layer l, network net)
{
    int b,i,k;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }

   /*
   int b,i,j,k,m,n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    float sum = 0;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    sum = 0;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            sum += net.input[index];
                        }
                    }
                    l.output[out_index] = sum/(l.size*l.size);

                }
            }
        }
    }
    */
}


layer make_avgpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*) calloc(output_size, sizeof(int));
    l.output =  (data_t*)calloc(output_size, sizeof(data_t));
    

    l.forward = forward_avgpool_layer;

    fprintf(stderr, "avg          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    
/// return param
    l.params.h = l.out_h;
    l.params.w = l.out_w;
    l.params.c = l.out_c;
    l.params.inputs = l.outputs;
    l.params.batch = batch;

    return l;
}

layer parse_avgpool(int size, int stride, int padding, size_params params)
{
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    layer avgpool_layer = make_avgpool_layer(batch,h,w,c,size,stride,padding);
    return avgpool_layer;
}
