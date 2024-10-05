#include "viz.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <raylib.h>
#include "configs.h"
#include "nn.h"
#include "train.h"

#define SPACING 1.0
#define COLOR_WEIGHTS WHITE
#define COLOR_ACTIVATION ORANGE
#define COLOR_GRAD GREEN

Camera3D g_camera3d;
float g_cam_angle = 0;
MnistRecord g_img_input;
Flags g_flags;
Thresholds g_thresholds;
Net g_net;

// Init visualization
bool viz_init()
{
    // Initialize Raylib window
    InitWindow(WINDOW_W, WINDOW_H, "NN");
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(FPS);

    // Load neural network
    if (!net_load(&g_net))
    {
        return true;
    }

    // Camera setup
    reset_cam();

    // Set flag defaults
    g_flags.cam_rotate = true;
    g_flags.draw_connections = true;
    g_flags.draw_cubes = true;
    g_flags.draw_cube_lines = true;
    g_flags.load_test_imgs = true;

    // Set threshold defaults
    g_thresholds.activations = 25;
    g_thresholds.connections = 25;
    g_thresholds.weight_cloud = 50;

    return false;
}

// Deinit visualization
void viz_deinit()
{
    net_free(&g_net);
    CloseWindow();
}

// Check for window close
bool is_viz_terminate()
{
    return WindowShouldClose();
}

// Update visualization logic
void viz_update(MnistRecord *test_img)
{
    // Handle keyboard inputs
    handle_keyboard_input();

    // Update camera position if rotating
    if (g_flags.cam_rotate)
    {
        g_cam_angle += CAM_REVOLUTION_SPEED * GetFrameTime();
        g_camera3d.position.x = cos(g_cam_angle) * CAM_REVOLUTION_RADIUS;
        g_camera3d.position.z = sin(g_cam_angle) * CAM_REVOLUTION_RADIUS;
    }
    else
    {
        UpdateCamera(&g_camera3d, CAMERA_FREE);
    }

    // Load test image if flag is set
    if (g_flags.load_test_imgs)
    {
        memcpy(g_img_input.pixels, test_img->pixels, sizeof(test_img->pixels));
    }

    // Initialize gradient and contribution networks
    Net grad_net = {};
    Net contrib_net = {};
    net_init_mem(&grad_net, true);
    net_init_mem(&contrib_net, true);

    // Run inference, calculate gradients, and activations
    float **activations;
    float loss = net_backward(&g_net, &g_img_input, &grad_net, &contrib_net, false);
    activations = net_forward(&g_net, &g_img_input, &contrib_net, false);
    float *preds = activations[len(activations) - 1];
    int prediction_idx = get_prediction_index(preds);

    // Prepare visualization data
    for (int i = 0; i < len(grad_net.layers); i++)
    {
        normalize_values(grad_net.layers[i].w[0], len(grad_net.layers[i].w[0]));
    }

    // Draw the 3D and 2D visualizations
    BeginDrawing();
    ClearBackground(BLUE);
    draw_3d(activations, &grad_net, &contrib_net, prediction_idx);
    draw_2d(prediction_idx, preds);
    EndDrawing();
}

// Reset camera settings
void reset_cam()
{
    g_camera3d.position = (Vector3){0, 30, 0};
    g_camera3d.target = (Vector3){0, 0, 0};
    g_camera3d.up = (Vector3){0, 1, 0};
    g_camera3d.fovy = 75.0f;
    g_camera3d.projection = CAMERA_PERSPECTIVE;
    g_cam_angle = 0;
}

// Draw 2D visualization elements (e.g. prediction graph, input grid)
void draw_2d(int pred_idx, float *preds)
{
    const int PADDING = 30;
    const int GRAPH_HEIGHT = 100;

    int input_grid_width = MNIST_IMG_SIZE * 8;
    int input_grid_start_y = GetRenderHeight() - input_grid_width - PADDING;

    // Draw GUI and bar graph
    draw_bar_graph(preds, PADDING, input_grid_start_y - GRAPH_HEIGHT, input_grid_width + PADDING, GRAPH_HEIGHT);
    draw_2d_image_input_grid(PADDING, input_grid_start_y);
    DrawFPS(GetRenderWidth() - 100, PADDING);
}

// Draw 3D visualization elements
void draw_3d(float ***activations, Net *grads, Net *contribs, int prediction_idx)
{
    BeginMode3D(g_camera3d);

    Shape *shapes = NULL;     // Dynamic array of shapes
    SceneObject *objs = NULL; // Dynamic array of scene objects

    // Collect 3D shapes
    collect_3d_shapes(&shapes, grads, contribs, activations, prediction_idx);

    // Sort and draw shapes
    for (int i = 0; i < len(shapes); i++)
    {
        if (shapes[i].cube.pos.x > 0)
        {
            DrawCube(shapes[i].cube.pos, SPACING, SPACING, SPACING, shapes[i].cube.color);
        }
        if (shapes[i].line.start.x > 0)
        {
            DrawLine3D(shapes[i].line.start, shapes[i].line.end, shapes[i].line.color);
        }
        if (shapes[i].cuboid.pos.x > 0)
        {
            DrawCubeWiresV(shapes[i].cuboid.pos, shapes[i].cuboid.size, shapes[i].cuboid.color);
        }
    }

    EndMode3D();
}

// Collect 3D shapes for visualization
void collect_3d_shapes(Shape **shapes, Net *grads, Net *contribs, float ***activations, int prediction_idx)
{
    // Implementation based on layers, gradients, and contributions (converted from Odin)
    // Here, shapes array would be filled with Cube, Line, and Cuboid based on layers

    // Placeholder for logic:
    for (int i = 0; i < sizeof(grads->layers); i++)
    {
        Shape new_shape = {};
        // Fill new_shape with the cube/line/cuboid information
        // Then append the shape to the shapes array
        *shapes = realloc(*shapes, (i + 1) * sizeof(Shape));
        (*shapes)[i] = new_shape;
    }
}

// Handle keyboard inputs
void handle_keyboard_input()
{
    if (IsKeyPressed(KEY_SPACE))
    {
        g_flags.cam_rotate = !g_flags.cam_rotate;
    }
    if (IsKeyPressed(KEY_TAB))
    {
        g_flags.draw_connections = !g_flags.draw_connections;
    }
    if (IsKeyPressed(KEY_R))
    {
        memset(&g_img_input, 0, sizeof(g_img_input));
    }
}

// Normalize values for activations and gradients
void normalize_values(float *values, int len)
{
    float sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += values[i];
    }

    for (int i = 0; i < len; i++)
    {
        values[i] /= sum;
    }
}

// Calculate squared distance between two vectors
float calc_vec3_dist_squared(Vector3 a, Vector3 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Map threshold values for rendering
float map_threshold_value(float value, float x, float y)
{
    return x + ((value / 100.0f) * (y - x));
}