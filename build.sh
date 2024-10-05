#!/bin/bash

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

if [ ! -f "Makefile" ]; then
    cmake ..
fi

make

./main