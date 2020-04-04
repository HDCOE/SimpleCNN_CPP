


//using half_float::half;

/*
typedef ap_fixed<32,8> fix16;
typedef ap_fixed<32,8>  data_t;
*/

typedef ap_fixed<16,5> fix16;
typedef float  data_t;


//#include "timer.h"
#include "utilyolo.h"
#include "layer.h"

#include "network.h"
#include "activation.h"
#include "conv.h"
#include "pool.h"
#include "avgpool.h"
#include "softmax.h"
#include "load_pretrain.h"
#include "detect.h"
#include "network_box.h"
#include "FC_layer.h"
#include "NMS.h"
#include "drawbox.h"

network *make_network(int n);
layer get_network_output_layer(network *net);
