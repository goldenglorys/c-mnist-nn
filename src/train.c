#include "train.h"
#include "nn.h"
#include "configs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Load MNIST Data from CSV
MnistRecord *load_mnist_data(const char *path, int size)
{
    FILE *file = fopen(path, "r");
    if (!file)
    {
        printf("Failed to open file: %s\n", path);
        return NULL;
    }

    char line[4096];                 // Buffer for each line
    fgets(line, sizeof(line), file); // Skip header line

    MnistRecord *data = (MnistRecord *)malloc(size * sizeof(MnistRecord));

    for (int i = 0; i < size; i++)
    {
        if (!fgets(line, sizeof(line), file))
            break;

        // Split the line by commas
        char *token = strtok(line, ",");
        data[i].label = (uint8_t)atoi(token); // First token is the label

        int pixel_index = 0;
        while (token != NULL && pixel_index < MNIST_IMG_DATA_LEN)
        {
            token = strtok(NULL, ",");
            if (token)
            {
                data[i].pixels[pixel_index++] = atof(token) / 255.0f; // Normalize pixel value
            }
        }
    }

    fclose(file);
    return data;
}

// Perform one training step
float train_step(Net *net, MnistRecord *batch, int batch_size, float learning_rate)
{
    Net grad = {}; // Temporary structure to hold gradients
    net_init_mem(&grad, true);

    float total_loss = 0.0f;

    // Backpropagate and accumulate gradients for the entire batch
    for (int i = 0; i < batch_size; i++)
    {
        float loss = net_backward(net, &batch[i], &grad, NULL, true);
        total_loss += loss;
    }

    // Update weights and biases based on averaged gradients
    for (int i = 0; i < sizeof(NET_ARCH) / sizeof(NET_ARCH[0]) + 1; i++)
    { // Dynamically determine the number of layers
        Layer *net_layer = &net->layers[i];
        Layer *grad_layer = &grad.layers[i];

        for (int j = 0; j < sizeof(net_layer->b) / sizeof(float); j++)
        {
            net_layer->b[j] -= learning_rate * grad_layer->b[j] / batch_size;
            for (int k = 0; k < sizeof(net_layer->w[j]) / sizeof(float); k++)
            {
                net_layer->w[j][k] -= learning_rate * grad_layer->w[j][k] / batch_size;
            }
        }
    }

    net_free(&grad);
    return total_loss / batch_size;
}

// Train the neural network
void train()
{
    srand((unsigned int)time(NULL));

    printf("Loading Training data ...\n");
    MnistRecord *train_data = load_mnist_data(MNIST_TRAIN_FILE_PATH, TRAIN_DATA_LEN);
    if (!train_data)
        return;

    printf("Loading Testing data ...\n");
    MnistRecord *test_data = load_mnist_data(MNIST_TEST_FILE_PATH, TEST_DATA_LEN);
    if (!test_data)
        return;

    // Augment training data
    printf("Augmenting training data ...\n");
    int data_len = TRAIN_DATA_LEN;
    for (int n = 0; n <= DATA_AUGMENTATION_COUNT; n++)
    {
        for (int i = data_len - 1; i >= 0; i--)
        {
            MnistRecord augmented = augment_mnist_record(&train_data[i]);
            data_len++;
            train_data = (MnistRecord *)realloc(train_data, data_len * sizeof(MnistRecord));
            train_data[data_len - 1] = augmented;
        }
    }
    printf("Done Augmenting, train data len: %d, test data len: %d\n", data_len, TEST_DATA_LEN);

    // Initialize neural network
    Net net = {};
    net_init_mem(&net, false);
    net_init_values(&net);

    // Training loop
    int batch_start = 0;
    int steps = NUM_STEPS;
    float learning_rate = LEARNING_RATE;

    for (int step = 0; step < steps; step++)
    {
        MnistRecord *batch = &train_data[batch_start];
        batch_start = (batch_start + BATCH_SIZE) % data_len;

        float loss = train_step(&net, batch, BATCH_SIZE, learning_rate);

        // Every 250 steps, print accuracy and learning rate
        if (step % 250 == 0)
        {
            float accuracy = calc_net_accuracy(test_data, &net);
            printf("Step: %d, Accuracy: %.4f, Loss: %.4f, Learning Rate: %.4f\n", step, accuracy, loss, learning_rate);
        }

        // Every 2500 steps, save the network
        if (step % 2500 == 0)
        {
            if (!net_save(&net))
            {
                printf("Failed to save network\n");
            }
        }
    }

    net_free(&net);
    free(train_data);
    free(test_data);
}

// Calculate the network's accuracy on a test dataset
float calc_net_accuracy(MnistRecord *test_dataset, Net *net)
{
    int correct_count = 0;

    for (int i = 0; i < TEST_DATA_LEN; i++)
    {
        float **predictions = net_forward(net, &test_dataset[i], NULL, false);
        int predicted_label = get_prediction_index(predictions[sizeof(NET_ARCH) / sizeof(NET_ARCH[0])]); // Dynamically determine last layer

        if (predicted_label == test_dataset[i].label)
        {
            correct_count++;
        }

        free(predictions); // Free predictions after each iteration
    }

    return (float)correct_count / TEST_DATA_LEN;
}

// Augment an MNIST record (rotate, shift)
MnistRecord augment_mnist_record(MnistRecord *record)
{
    MnistRecord augmented;
    augmented.label = record->label;

    int augmentation = rand() % 2;
    if (augmentation == 0)
    {
        float angle = ((float)rand() / RAND_MAX) * 30.0f - 15.0f;
        memcpy(augmented.pixels, rotate_image(record->pixels, angle), MNIST_IMG_DATA_LEN * sizeof(float));
    }
    else
    {
        int dx = (rand() % 7) - 3;
        int dy = (rand() % 7) - 3;
        memcpy(augmented.pixels, shift_image(record->pixels, dx, dy), MNIST_IMG_DATA_LEN * sizeof(float));
    }

    return augmented;
}

// Rotate an image by a given angle
float *rotate_image(float *pixels, float angle)
{
    static float result[MNIST_IMG_DATA_LEN];
    memset(result, 0, MNIST_IMG_DATA_LEN * sizeof(float));

    float center = (float)MNIST_IMG_SIZE / 2.0f;
    float radians = angle * M_PI / 180.0f;
    float cos_a = cosf(radians);
    float sin_a = sinf(radians);

    for (int y = 0; y < MNIST_IMG_SIZE; y++)
    {
        for (int x = 0; x < MNIST_IMG_SIZE; x++)
        {
            int new_x = (int)((x - center) * cos_a - (y - center) * sin_a + center);
            int new_y = (int)((x - center) * sin_a + (y - center) * cos_a + center);

            if (new_x >= 0 && new_x < MNIST_IMG_SIZE && new_y >= 0 && new_y < MNIST_IMG_SIZE)
            {
                result[y * MNIST_IMG_SIZE + x] = pixels[new_y * MNIST_IMG_SIZE + new_x];
            }
        }
    }

    return result;
}

// Shift an image by a given dx and dy
float *shift_image(float *pixels, int dx, int dy)
{
    static float result[MNIST_IMG_DATA_LEN];
    memset(result, 0, MNIST_IMG_DATA_LEN * sizeof(float));

    for (int y = 0; y < MNIST_IMG_SIZE; y++)
    {
        for (int x = 0; x < MNIST_IMG_SIZE; x++)
        {
            int new_x = x + dx;
            int new_y = y + dy;

            if (new_x >= 0 && new_x < MNIST_IMG_SIZE && new_y >= 0 && new_y < MNIST_IMG_SIZE)
            {
                result[y * MNIST_IMG_SIZE + x] = pixels[new_y * MNIST_IMG_SIZE + new_x];
            }
        }
    }

    return result;
}

// Get the predicted index from softmax outputs
int get_prediction_index(float *preds)
{
    int max_idx = 0;
    for (int i = 1; i < MNIST_NUM_LABELS; i++)
    {
        if (preds[i] > preds[max_idx])
        {
            max_idx = i;
        }
    }
    return max_idx;
}
