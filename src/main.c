#include <raylib.h>

int main(void)
{
    InitWindow(800, 600, "raylib viz");
    SetTargetFPS(60);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText("system check raylib..", 190, 200, 20, LIGHTGRAY);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}