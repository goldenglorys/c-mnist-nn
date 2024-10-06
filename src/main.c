#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "viz.h"
#include "train.h"
#include "nn.h"
#include "raylib.h"

// Function prototypes
void check_memory_allocations(void); // Now matches the required prototype
void run_viz();

// Memory tracking structure (simplified)
typedef struct
{
    size_t size;
    void *location;
} AllocationEntry;

typedef struct
{
    AllocationEntry *allocation_map;
    size_t allocation_count;
    size_t *bad_free_array;
    size_t bad_free_count;
} TrackingAllocator;

TrackingAllocator track; // Global variable for memory tracking

// Mock of memory tracking (in lieu of Odin's tracking_allocator)
void tracking_allocator_init(TrackingAllocator *track)
{
    track->allocation_map = malloc(1024 * sizeof(AllocationEntry)); // Example allocation map size
    track->allocation_count = 0;
    track->bad_free_array = malloc(1024 * sizeof(size_t));
    track->bad_free_count = 0;
}

void tracking_allocator_destroy(TrackingAllocator *track)
{
    free(track->allocation_map);
    free(track->bad_free_array);
}

void check_memory_allocations(void)
{ // No parameters now, to be compatible with atexit()
    if (track.allocation_count > 0)
    {
        printf("=== %zu allocations not freed: ===\n", track.allocation_count);
        for (size_t i = 0; i < track.allocation_count; i++)
        {
            printf("- %zu bytes @ %p\n", track.allocation_map[i].size, track.allocation_map[i].location);
        }
    }
    if (track.bad_free_count > 0)
    {
        printf("=== %zu incorrect frees: ===\n", track.bad_free_count);
        for (size_t i = 0; i < track.bad_free_count; i++)
        {
            printf("- %p was incorrectly freed\n", (void *)track.bad_free_array[i]);
        }
    }
    tracking_allocator_destroy(&track); // Use the global `track`
}

// Seed random number generator with current time in nanoseconds
void seed_random()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand((unsigned int)ts.tv_nsec); // Seeding with nanoseconds
}

// Run the visualization loop
void run_viz()
{
    // Viz Init
    if (viz_init())
    {
        return; // Exit if initialization failed
    }
    atexit(viz_deinit); // Ensure cleanup happens on exit

    // Load test data
    MnistRecord *test_data = load_mnist_data(MNIST_TEST_FILE_PATH, TEST_DATA_LEN);
    if (!test_data)
    {
        printf("Failed to read MNIST test data file\n");
        return;
    }

    int frame_idx = 0;
    int img_idx = 0;

    // Simulation loop
    while (!is_viz_terminate())
    {
        frame_idx = (frame_idx + 1) % 15;
        if (frame_idx == 0)
        {
            img_idx = (img_idx + 1) % TEST_DATA_LEN;
        }

        viz_update(&test_data[img_idx]);

        // Simulating memory cleanup (free temporary allocations)
        // free_all(context.temp_allocator);  // Assuming you handle this elsewhere
    }

    free(test_data); // Free test data after the loop
}

// Main function
int main()
{
    // Memory tracking allocator setup
    tracking_allocator_init(&track);
    atexit(check_memory_allocations); // Register global memory check at program exit

    // Seed the random number generator
    seed_random();

    // Run the visualization (or training if desired)
    run_viz();

    return 0;
}