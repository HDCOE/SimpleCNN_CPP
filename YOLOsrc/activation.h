

static inline data_t linear_activate(data_t x){return x;}
static inline data_t logistic_activate(data_t x){return (data_t)(1./(1. + exp((float)(-x))));}
static inline data_t relu_activate(data_t x){return (data_t)(x*(x>0));}
static inline data_t leaky_activate(data_t x){return (x>0) ? (data_t)x : (data_t)((.1*(float)x));}
static inline data_t tanh_activate(data_t x){return (data_t)((exp(2*(float)x)-1)/(exp(2*(float)x)+1));}
data_t activate(data_t x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case RELU:
            return relu_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
    }
    return (data_t)0;
}

void activate_array(data_t *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}