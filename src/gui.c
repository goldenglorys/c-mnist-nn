#include "gui.h"
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECKBOX_SIZE 20
#define BUTTON_PADDING 15
#define FONT_SIZE 20
#define WIDGET_HEIGHT 20
#define TAB_HEIGHT 30
#define TAB_LEFT_PADDING 30

void ui_dropdown_list(DropdownList *menu);
void ui_container(Container *container);
void ui_button(Button *button, int posx, int posy);
void ui_checkbox(CheckBox *checkbox, int posx, int posy);
void ui_slider(Slider *slider, int posx, int posy);
float clamp(float value, float min, float max);
int calculate_container_height(Container *container);

void show_gui(bool *flags, float *thresholds, void (*reset_cam)())
{
    static int active_menu_index = -1;
    static bool slider_updates[3] = {false, false, false};

    Container controls = {
        .label = "Controls",
        .widgets = (Widget[]){
            {.checkbox = {"Rotate Cam", &flags[0]}},
            {.checkbox = {"Show Cube Lines", &flags[1]}},
            {.checkbox = {"Show Cubes", &flags[2]}},
            {.checkbox = {"Show Weight Cloud", &flags[3]}},
            {.checkbox = {"Show Node Activations", &flags[4]}},
            {.checkbox = {"Show Connections", &flags[5]}},
            {.checkbox = {"Load Test Images", &flags[6]}},
            {.button = {"Reset Camera", reset_cam}}},
    };

    Container thresholds_container = {
        .label = "Thresholds",
        .widgets = (Widget[]){
            {.slider = {"Weight Cloud", 0, 100, &thresholds[0], &slider_updates[0]}},
            {.slider = {"Activation", 0, 100, &thresholds[1], &slider_updates[1]}},
            {.slider = {"Connection", 0, 100, &thresholds[2], &slider_updates[2]}},
        },
    };

    DropdownList menu = {
        .pos = {20, 20},
        .containers = (Container[]){controls, thresholds_container},
        .active_menu_index = &active_menu_index,
    };

    ui_dropdown_list(&menu);
}

void ui_dropdown_list(DropdownList *menu)
{
    float total_height = 0;
    for (int i = 0; i < 2; i++)
    {
        Container *container = &menu->containers[i];
        bool is_active = i == *menu->active_menu_index;
        char label_fmt[10];
        sprintf(label_fmt, "%s %s", is_active ? "<" : ">", container->label);
        int text_width = MeasureText(label_fmt, FONT_SIZE);

        DrawText(label_fmt, menu->pos.x + 5, menu->pos.y + total_height + 5, FONT_SIZE, WHITE);

        bool is_mouse_on_area = CheckCollisionPointRec(GetMousePosition(), (Rectangle){menu->pos.x, menu->pos.y + total_height, text_width + 10, TAB_HEIGHT});
        if (is_mouse_on_area && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {
            *menu->active_menu_index = is_active ? -1 : i;
        }

        total_height += TAB_HEIGHT;
        if (is_active)
        {
            container->pos = (Vector2){menu->pos.x + TAB_LEFT_PADDING, menu->pos.y + total_height + WIDGET_HEIGHT};
            ui_container(container);
            int height = calculate_container_height(container);
            total_height += height + WIDGET_HEIGHT * 2;
        }
    }
}

void ui_container(Container *container)
{
    int posx = container->pos.x;
    int posy = container->pos.y;
    for (int i = 0; i < 8; i++)
    {
        Widget *w = &container->widgets[i];
        switch (i)
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            ui_checkbox(&w->checkbox, posx, posy);
            posy += WIDGET_HEIGHT / 2;
            break;
        case 7:
            posy += 5;
            ui_button(&w->button, posx, posy);
            posy += WIDGET_HEIGHT;
            break;
        case 8:
        case 9:
        case 10:
            ui_slider(&w->slider, posx, posy);
            posy += WIDGET_HEIGHT * 2;
            break;
        }
        posy += WIDGET_HEIGHT;
    }
}

void ui_button(Button *button, int posx, int posy)
{
    posx += BUTTON_PADDING / 2;
    int text_width = MeasureText(button->label, FONT_SIZE);
    float total_width = text_width + BUTTON_PADDING;
    float total_height = FONT_SIZE + BUTTON_PADDING;
    bool is_mouse_on_area = CheckCollisionPointRec(GetMousePosition(), (Rectangle){posx, posy, total_width, total_height});

    if (is_mouse_on_area)
    {
        DrawRectangle(posx - BUTTON_PADDING / 2, posy - BUTTON_PADDING / 2, total_width, total_height, Fade(GRAY, 0.5));
    }
    DrawRectangleLines(posx - BUTTON_PADDING / 2, posy - BUTTON_PADDING / 2, total_width, total_height, WHITE);
    DrawText(button->label, posx, posy, FONT_SIZE, WHITE);

    if (is_mouse_on_area && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        button->on_click();
    }
}

void ui_checkbox(CheckBox *checkbox, int posx, int posy)
{
    int checkbox_enabled_size = CHECKBOX_SIZE - 6;
    int enabled_size_diff = (CHECKBOX_SIZE - checkbox_enabled_size) / 2;
    int text_width = MeasureText(checkbox->label, FONT_SIZE);

    DrawRectangleLines(posx, posy, CHECKBOX_SIZE, CHECKBOX_SIZE, WHITE);
    if (*checkbox->is_enabled)
    {
        DrawRectangle(posx + enabled_size_diff, posy + enabled_size_diff, checkbox_enabled_size, checkbox_enabled_size, WHITE);
    }

    int text_pos_x = posx + CHECKBOX_SIZE + 10;
    int text_pos_y = posy + CHECKBOX_SIZE / 2 - 10;
    DrawText(checkbox->label, text_pos_x, text_pos_y, FONT_SIZE, WHITE);

    float total_width = CHECKBOX_SIZE + 10 + text_width;
    float total_height = max(CHECKBOX_SIZE, FONT_SIZE);
    bool is_mouse_on_area = CheckCollisionPointRec(GetMousePosition(), (Rectangle){posx, posy, total_width, total_height});
    if (is_mouse_on_area && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        *checkbox->is_enabled = !(*checkbox->is_enabled);
    }
}

void ui_slider(Slider *slider, int posx, int posy)
{
    const int SLIDER_W = 200;
    const int SLIDER_H = 20;
    const int LABEL_HEIGHT = 20;
    const int KNOB_SIZE = SLIDER_H;
    const int VALUE_WIDTH = 50;

    char label[100];
    sprintf(label, "%s - %.1f", slider->label, *slider->value);

    DrawText(label, posx, posy, FONT_SIZE, WHITE);

    int slider_y = posy + LABEL_HEIGHT;
    int slider_end_x = posx + SLIDER_W;

    DrawRectangleLines(posx, slider_y, SLIDER_W, SLIDER_H, WHITE);

    int knob_pos_x = posx + (int)((*slider->value - slider->min) / (slider->max - slider->min) * (SLIDER_W - KNOB_SIZE));
    knob_pos_x = (int)clamp(knob_pos_x, posx, slider_end_x - KNOB_SIZE);
    int knob_pos_y = slider_y + SLIDER_H / 2 - KNOB_SIZE / 2;

    DrawRectangle(posx, slider_y, knob_pos_x - posx + KNOB_SIZE / 2, SLIDER_H, WHITE);
    DrawRectangle(knob_pos_x, knob_pos_y, KNOB_SIZE, KNOB_SIZE, WHITE);

    Vector2 mouse_pos = GetMousePosition();
    Rectangle slider_rect = {posx, slider_y, SLIDER_W, SLIDER_H};

    if (CheckCollisionPointRec(mouse_pos, slider_rect) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        *slider->is_updating = true;
    }

    if (*slider->is_updating)
    {
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON))
        {
            float new_value = slider->min + (slider->max - slider->min) * clamp(mouse_pos.x - posx, 0, SLIDER_W) / SLIDER_W;
            *slider->value = clamp(new_value, slider->min, slider->max);
        }
        else
        {
            *slider->is_updating = false;
        }
    }
}

float clamp(float value, float min, float max)
{
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

int calculate_container_height(Container *container)
{
    int height = 0;
    for (int i = 0; i < 8; i++)
    {
        Widget *w = &container->widgets[i];
        switch (i)
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            height += WIDGET_HEIGHT + WIDGET_HEIGHT / 2;
            break;
        case 7:
            height += WIDGET_HEIGHT;
            break;
        case 8:
        case 9:
        case 10:
            height += WIDGET_HEIGHT * 3;
            break;
        }
    }
    return height;
}