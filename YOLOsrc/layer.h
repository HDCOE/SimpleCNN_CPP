typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
  //  network *net;
} size_params;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    size_params params; 
    void (*forward)   (struct layer, struct network);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    data_t smooth;
    data_t dot;
    data_t angle;
    data_t jitter;
    data_t saturation;
    data_t exposure;
    data_t shift;
    data_t ratio;
    data_t learning_rate_scale;
    data_t clip;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    data_t alpha;
    data_t beta;
    data_t kappa;

    data_t coord_scale;
    data_t object_scale;
    data_t noobject_scale;
    data_t mask_scale;
    data_t class_scale;
    int bias_match;
    int random;
    data_t ignore_thresh;
    data_t truth_thresh;
    data_t thresh;
    data_t focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
   // int dontload;
    int dontsave;
  //  int dontloadscales;

    data_t temperature;
    data_t probability;
    data_t scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    data_t * rand;
    data_t * cost;
    data_t * state;
    data_t * prev_state;
    data_t * forgot_state;
    data_t * forgot_delta;
    data_t * state_delta;
    data_t * combine_cpu;
    data_t * combine_delta_cpu;

    data_t * concat;
    data_t * concat_delta;

    data_t * binary_weights;

    data_t * biases;
    data_t * bias_updates;

    data_t * scales;
    data_t * scale_updates;

    data_t * weights;
    data_t * weight_updates;

    data_t * delta;
    data_t * output;
    data_t * loss;
    data_t * squared;
    data_t * norms;

    data_t * spatial_mean;
    data_t * mean;
    data_t * variance;

    data_t * mean_delta;
    data_t * variance_delta;

    data_t * rolling_mean;
    data_t * rolling_variance;

    data_t * x;
    data_t * x_norm;

    data_t * m;
    data_t * v;
    
    data_t * bias_m;
    data_t * bias_v;
    data_t * scale_m;
    data_t * scale_v;


    data_t *z_cpu;
    data_t *r_cpu;
    data_t *h_cpu;
    data_t * prev_state_cpu;

    data_t *temp_cpu;
    data_t *temp2_cpu;
    data_t *temp3_cpu;

    data_t *dh_cpu;
    data_t *hh_cpu;
    data_t *prev_cell_cpu;
    data_t *cell_cpu;
    data_t *f_cpu;
    data_t *i_cpu;
    data_t *g_cpu;
    data_t *o_cpu;
    data_t *c_cpu;
    data_t *dc_cpu; 

    data_t * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
    
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    //tree *softmax_tree;
    size_t workspace_size;
};
void free_layer(layer l)
{
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
}
