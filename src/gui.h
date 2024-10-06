#ifndef GUI_H
#define GUI_H

#include <stdbool.h>
#include <raylib.h>

typedef struct
{
    const char *label;
    bool *is_enabled;
} CheckBox;

typedef struct
{
    const char *label;
    void (*on_click)();
} Button;

typedef struct
{
    const char *label;
    float min, max;
    float *value;
    bool *is_updating;
} Slider;

typedef union
{
    CheckBox checkbox;
    Button button;
    Slider slider;
} Widget;

typedef struct
{
    const char *label;
    Vector2 pos;
    Widget *widgets;
    bool is_visible;
} Container;

typedef struct
{
    Container *containers;
    int *active_menu_index;
    Vector2 pos;
} DropdownList;

void show_gui(bool *flags, float *thresholds, void (*reset_cam)());

#define max(a,b) ((a) > (b) ? (a) : (b))

#endif