#ifndef VIZ_H
#define VIZ_H

#include <stdbool.h>
#include <raylib.h>
#include "nn.h"

typedef struct
{
    bool cam_rotate;
    bool draw_connections;
    bool draw_cubes;
    bool draw_cube_lines;
    bool draw_node_activations;
    bool draw_weight_cloud;
    bool load_test_imgs;
} Flags;

typedef struct
{
    float weight_cloud;
    float activations;
    float connections;
} Thresholds;

typedef struct
{
    int index;
    float **weights;
    float **grads;
    float **contribs;
    float z_offset;
    int rows, columns, depth;
    Color grid_color;
} LayerViz;

typedef struct
{
    Vector3 pos;
    Color color;
    int layer_index;
} Cube;

typedef struct
{
    Vector3 start;
    Vector3 end;
    Color color;
} Line;

typedef struct
{
    Vector3 pos;
    Vector3 size;
    Color color;
    bool is_activated;
} Cuboid;

typedef union
{
    Cube cube;
    Line line;
    Cuboid cuboid;
} Shape;

typedef struct
{
    Shape shape;
    float dist_to_cam;
} SceneObject;

bool viz_init();
void viz_deinit();
bool is_viz_terminate();
void viz_update(MnistRecord *test_img);
void reset_cam();

#endif