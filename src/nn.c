#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "nn.h"
#include "configs.h"

// Helper functions for random values
static float get_rand_bias()
{
    // Bias is scaled down to stabilize gradients at the start
    // Bias output in range [0.0, 1.0) * 0.01
    return ((float)rand() / RAND_MAX) * 0.01f;
}

static float get_rand_weight(float num_inputs, float num_nodes)
{
    // Xavier/Glorot initialization
    float limit = sqrt(6.0f / (num_inputs + num_nodes));
    float range = limit * 2;
    return ((float)rand() / RAND_MAX) * range - limit;
}

// ReLU activation function
static void relu(float *values, int len)
{
    for (int i = 0; i < len; i++)
    {
        values[i] = fmaxf(0, values[i]);
    }
}

// ReLU derivative (for backpropagation)
static void relu_derivative(float *values, int len)
{
    for (int i = 0; i < len; i++)
    {
        values[i] = values[i] > 0 ? 1 : 0;
    }
}

// Softmax activation function
static void softmax(float *values, int len)
{
    assert(len > 0);

    float max_val = values[0];
    for (int i = 1; i < len; i++)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    for (int i = 0; i < len; i++)
    {
        values[i] /= sum;
    }
}

// Initialize the network memory (weights and biases)
void net_init_mem(Net *net, bool use_temp_allocator)
{
    // Initialize the layers with sizes from NET_ARCH and add an output layer
    // int arch[] = NET_ARCH;
    int num_layers = sizeof(NET_ARCH) / sizeof(NET_ARCH[0]) + 1; // Output layer added
    int img_size = MNIST_IMG_DATA_LEN;

    net->layers = (Layer *)malloc(num_layers * sizeof(Layer));

    for (int i = 0; i < num_layers; i++)
    {
        int num_nodes = (i == num_layers - 1) ? MNIST_NUM_LABELS : NET_ARCH[i];
        int num_inputs = (i == 0) ? img_size : NET_ARCH[i - 1];

        // Initialize weights and biases
        net->layers[i].w = (float **)malloc(num_nodes * sizeof(float *));
        net->layers[i].b = (float *)malloc(num_nodes * sizeof(float));

        for (int j = 0; j < num_nodes; j++)
        {
            net->layers[i].w[j] = (float *)malloc(num_inputs * sizeof(float));
        }

        net->layers[i].activation = (i == num_layers - 1) ? SOFTMAX : RELU;
        net->layers[i].dropout_rate = 0;
    }
}

// Initialize the values of weights and biases using Xavier initialization
void net_init_values(Net *net)
{
    // int arch[] = NET_ARCH;
    int num_layers = sizeof(NET_ARCH) / sizeof(NET_ARCH[0]) + 1;
    int img_size = MNIST_IMG_DATA_LEN;

    for (int i = 0; i < num_layers; i++)
    {
        int num_nodes = (i == num_layers - 1) ? MNIST_NUM_LABELS : NET_ARCH[i];
        int num_inputs = (i == 0) ? img_size : NET_ARCH[i - 1];

        for (int j = 0; j < num_nodes; j++)
        {
            net->layers[i].b[j] = get_rand_bias();
            for (int k = 0; k < num_inputs; k++)
            {
                net->layers[i].w[j][k] = get_rand_weight((float)num_inputs, (float)num_nodes);
            }
        }

        net->layers[i].dropout_rate = DROPOUT_RATE;
    }
}

// Forward pass for a single layer
static float *layer_forward(Layer *layer, float *input, int num_inputs, bool is_train)
{
    int num_outputs = sizeof(layer->b) / sizeof(layer->b[0]);
    float *output = (float *)malloc(num_outputs * sizeof(float));

    for (int i = 0; i < num_outputs; i++)
    {
        output[i] = layer->b[i];
        for (int j = 0; j < num_inputs; j++)
        {
            output[i] += layer->w[i][j] * input[j];
        }
    }

    // Apply activation
    if (layer->activation == RELU)
    {
        relu(output, num_outputs);
    }
    else if (layer->activation == SOFTMAX)
    {
        softmax(output, num_outputs);
    }

    // Dropout (during training)
    if (is_train && layer->dropout_rate > 0)
    {
        for (int i = 0; i < num_outputs; i++)
        {
            if ((float)rand() / RAND_MAX < layer->dropout_rate)
            {
                output[i] = 0;
            }
            else
            {
                output[i] /= (1 - layer->dropout_rate);
            }
        }
    }

    return output;
}

// Forward pass for the entire network
float **net_forward(Net *net, MnistRecord *img, Net *contribs, bool is_train)
{
    int num_layers = sizeof(NET_ARCH) / sizeof(NET_ARCH[0]) + 1;
    float **activations = (float **)malloc((num_layers + 1) * sizeof(float *));

    // Input layer
    activations[0] = img->pixels;

    // Pass through each layer
    for (int i = 0; i < num_layers; i++)
    {
        int num_inputs = (i == 0) ? MNIST_IMG_DATA_LEN : sizeof(net->layers[i - 1].b) / sizeof(float);
        activations[i + 1] = layer_forward(&net->layers[i], activations[i], num_inputs, is_train);
    }

    return activations;
}

// Backpropagation and loss calculation
float net_backward(Net *net, MnistRecord *img, Net *grad, Net *contribs, bool is_train)
{
    float **activations = net_forward(net, img, contribs, is_train);

    int num_outputs = MNIST_NUM_LABELS;
    float *output_error = (float *)malloc(num_outputs * sizeof(float));
    for (int i = 0; i < num_outputs; i++)
    {
        output_error[i] = activations[num_outputs][i];
        if (i == img->label)
        {
            output_error[i] -= 1;
        }
    }

    float loss = -logf(activations[num_outputs][img->label]);

    // Backpropagate error through layers
    for (int i = num_outputs - 1; i >= 0; i--)
    {
        Layer *layer = &net->layers[i];
        Layer *grad_layer = &grad->layers[i];
        float *prev_act = activations[i];

        for (int j = 0; j < num_outputs; j++)
        {
            grad_layer->b[j] += output_error[j];
            for (int k = 0; k < MNIST_IMG_DATA_LEN; k++)
            {
                grad_layer->w[j][k] += output_error[j] * prev_act[k];
            }
        }

        if (i == 0)
            continue;

        // Compute the error for the previous layer
        float *prev_error = (float *)malloc(sizeof(prev_act) * sizeof(float));
        for (int j = 0; j < sizeof(prev_act) / sizeof(float); j++)
        {
            for (int k = 0; k < sizeof(output_error) / sizeof(float); k++)
            {
                prev_error[j] += output_error[k] * layer->w[k][j];
            }
        }

        if (net->layers[i - 1].activation == RELU)
        {
            relu_derivative(prev_act, sizeof(prev_act) / sizeof(float));
            for (int j = 0; j < sizeof(prev_error) / sizeof(float); j++)
            {
                prev_error[j] *= prev_act[j];
            }
        }

        free(output_error);
        output_error = prev_error;
    }

    return loss;
}

// Free network resources
void net_free(Net *net)
{
    if (!net)
        return;

    for (int i = 0; i < sizeof(net->layers) / sizeof(Layer); i++)
    {
        Layer *layer = &net->layers[i];
        for (int j = 0; j < sizeof(layer->w) / sizeof(float *); j++)
        {
            free(layer->w[j]);
        }
        free(layer->w);
        free(layer->b);
    }
    free(net->layers);
}

// Save the network to a file
bool net_save(Net *net)
{
    FILE *file = fopen(NETWORK_SAVE_FILE_PATH, "wb");
    if (!file)
    {
        return false;
    }

    fwrite(net, sizeof(Net), 1, file);
    fclose(file);

    return true;
}

// Load the network from a file
bool net_load(Net *net)
{
    FILE *file = fopen(NETWORK_LOAD_FILE_PATH, "rb");
    if (!file)
    {
        return false;
    }

    fread(net, sizeof(Net), 1, file);
    fclose(file);

    return true;
}
