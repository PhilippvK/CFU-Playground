#ifndef CONV_CODEBOOKS_H
#define CONV_CODEBOOKS_H

#include <stdint.h>

// Auto-generated Conv2D layer codebooks header
// Indexed by execution order (input->output, reverse order)

#define NUM_CONV_LAYERS 9
#define MAX_CODEBOOK_SIZE 16

typedef struct {
    int8_t codebook[MAX_CODEBOOK_SIZE];
    uint8_t n_values;
    const char* name;
} ConvLayerCodebook;

static const ConvLayerCodebook conv_codebooks[NUM_CONV_LAYERS] = {
    { { -127, -126, -123, -121, -119, -117, -116, -115, -113, -109, -108, -107, -105, -102, -98, -97 }, 16, "model/conv2d/Conv2D" },
    { { -125, -102, -83, -68, -53, -36, -24, -10, 3, 14, 22, 37, 53, 81, 100, 127 }, 16, "model/conv2d_1/Conv2D" },
    { { -113, -88, -75, -60, -46, -37, -23, -12, 4, 12, 26, 38, 48, 65, 85, 117 }, 16, "model/conv2d_2/Conv2D" },
    { { -127, -100, -82, -71, -55, -44, -30, -21, -10, 6, 19, 27, 40, 59, 84, 121 }, 16, "model/conv2d_3/Conv2D" },
    { { -127, -37, 31, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 4, "model/conv2d_4/Conv2D" },
    { { -127, -103, -90, -71, -58, -40, -26, -5, 14, 29, 41, 58, 79, 92, 113, 127 }, 16, "model/conv2d_5/Conv2D" },
    { { -127, -51, 17, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 4, "model/conv2d_6/Conv2D" },
    { { -127, -46, 26, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 4, "model/conv2d_7/Conv2D" },
    { { -127, -74, -50, -11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 4, "model/conv2d_8/Conv2D" },
};

#endif // CONV_CODEBOOKS_H

// Indices (by execution order) of layers with input channels >= 32
#define NUM_CONV_ACCEL_LAYERS 0
