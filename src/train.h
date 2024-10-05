#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"

MnistRecord *load_mnist_data(const char *path, int size);
void train();
float calc_net_accuracy(MnistRecord *test_dataset, Net *net);
MnistRecord augment_mnist_record(MnistRecord *record);
float *rotate_image(float *pixels, float angle);
float *shift_image(float *pixels, int dx, int dy);
float *add_noise(float *pixels, float noise_level);
int get_prediction_index(float *preds);

#endif