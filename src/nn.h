#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <stdint.h>
#include "configs.h"

typedef enum
{
    RELU,
    SOFTMAX
} Activation;

typedef struct
{
    float **w;
    float *b;
    Activation activation;
    float dropout_rate;
} Layer;

typedef struct
{
    Layer *layers;
} Net;

typedef struct
{
    float pixels[MNIST_IMG_DATA_LEN];
    uint8_t label;
} MnistRecord;

void net_init_mem(Net *net, bool use_temp_allocator);
void net_init_values(Net *net);
float **net_forward(Net *net, MnistRecord *img, Net *contribs, bool is_train);
float net_backward(Net *net, MnistRecord *img, Net *grad, Net *contribs, bool is_train);
void net_free(Net *net);
bool net_save(Net *net);
bool net_load(Net *net);

#endif