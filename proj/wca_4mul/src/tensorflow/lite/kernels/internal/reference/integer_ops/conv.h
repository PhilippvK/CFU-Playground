/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.  Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include <stdio.h>
#include "cfu.h"
#include <signal.h>
#include <stdlib.h>
#include "conv_codebooks.h"
#include "perf.h"

#define PRINT_DEBUG
#define PRINT_DEBUG_MACS   // Comment this line to disable debug MAC prints

// Funct7 values
#define CFU_FUNCT7_SET_CODEBOOK_2    0x20
#define CFU_FUNCT7_SET_CODEBOOK_4    0x28
#define CFU_FUNCT7_SET_CODEBOOK_16   0x38
#define CFU_FUNCT7_PUSH_WEIGHTS      0x10
#define CFU_FUNCT7_ALU_MAC           0x40
#define CFU_FUNCT7_ALU_RST           0x48
#define CFU_FUNCT7_MAC_READ          0x50
#define CFU_FUNCT7_DEBUG_DUMP        0x52
#define  CFU_FUNCT7_MAC_READ_NO_RESET  0x54

static int conv2d_call_count = 0;

void print_bits(uint32_t val) {
        for(int i = 31; i >= 0; i--) {
           printf("%d", (val >> i) & 1);
           }
}



namespace tflite {
namespace reference_integer_ops {


void AccelConvPerChannel_16(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data
) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  static const int8_t model_conv2d_8_Conv2D[4] = {-127, -65, -21, 42};

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  int32_t start_cycle = 0;
  int32_t end_cycle = 0;
  int32_t cycle_count = 0;
  int32_t idx_count = 0;
  int pcount=0;


  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
        TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    const int groups = input_depth / filter_input_depth;
    TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int filters_per_group = output_depth / groups;
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int8_t* layer_codebook = conv_codebooks[conv2d_call_count].codebook;
    static bool print_once = true;
    int32_t count=0;
    uint32_t codebook_word0 = 
        ((uint32_t)(layer_codebook[3]  & 0xFF) << 24) |
        ((uint32_t)(layer_codebook[2]  & 0xFF) << 16) |
        ((uint32_t)(layer_codebook[1]  & 0xFF) << 8)  |
        ((uint32_t)(layer_codebook[0]  & 0xFF));

    uint32_t codebook_word1 = 
        ((uint32_t)(layer_codebook[7]  & 0xFF) << 24) |
        ((uint32_t)(layer_codebook[6]  & 0xFF) << 16) |
        ((uint32_t)(layer_codebook[5]  & 0xFF) << 8)  |
        ((uint32_t)(layer_codebook[4]  & 0xFF));

    uint32_t codebook_word2 = 
        ((uint32_t)(layer_codebook[11] & 0xFF) << 24) |
        ((uint32_t)(layer_codebook[10] & 0xFF) << 16) |
        ((uint32_t)(layer_codebook[9]  & 0xFF) << 8)  |
        ((uint32_t)(layer_codebook[8]  & 0xFF));

    uint32_t codebook_word3 = 
        ((uint32_t)(layer_codebook[15] & 0xFF) << 24) |
        ((uint32_t)(layer_codebook[14] & 0xFF) << 16) |
        ((uint32_t)(layer_codebook[13] & 0xFF) << 8)  |
        ((uint32_t)(layer_codebook[12] & 0xFF));

    // First 8 codebook values (0..7)
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_16, codebook_word0, codebook_word1);
    // Second 8 codebook values (8..15)
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_16, codebook_word2, codebook_word3);



    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    // Reset accumulator for new output channel
                    cfu_op0(CFU_FUNCT7_ALU_RST, 0, 0);
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            // Zero padding
                            if (in_x < 0 || in_x >= input_width || in_y < 0 || in_y >= input_height) {
                                continue;
                            }
                            const int outer_base = Offset(input_shape, batch, in_y, in_x, group * filter_input_depth);
                            const int8_t* act_base_ptr = input_data + outer_base;
                            for (int in_channel = 0; in_channel < filter_input_depth; in_channel += 16) {
                                const int8_t* base = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                                int32_t val132 = *((const int32_t*)base);
                                int32_t val232 = *((const int32_t*)(base + 4));
                                cfu_op0(CFU_FUNCT7_PUSH_WEIGHTS, val132, val232);
                                const uint32_t* act_words = (const uint32_t*)(act_base_ptr + in_channel);
                                cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[0], act_words[1]);
                                cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2], act_words[3]);
                            }
                        }
                    }
                    int32_t acc = cfu_op0(CFU_FUNCT7_MAC_READ, 0, 0);
                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int8_t>(acc);
          }
        }
      }
    }
  return;
}

void AccelConvPerChannel(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data
) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  static const int8_t model_conv2d_8_Conv2D[4] = {-127, -65, -21, 42};

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  int32_t start_cycle = 0;
  int32_t end_cycle = 0;
  int32_t cycle_count = 0;
  int32_t idx_count = 0;
  int pcount=0;


  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int8_t* layer_codebook = conv_codebooks[conv2d_call_count].codebook;
  static bool print_once = true;
  int32_t count=0;

  cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_4, 
    ((uint32_t)(layer_codebook[3] & 0xFF) << 24) |
    ((uint32_t)(layer_codebook[2] & 0xFF) << 16) |
    ((uint32_t)(layer_codebook[1] & 0xFF) << 8)  |
    ((uint32_t)(layer_codebook[0] & 0xFF)), 0);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            auto group = out_channel / filters_per_group;
            // Reset accumulator for new output channel
            cfu_op0(CFU_FUNCT7_ALU_RST, 0, 0);
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
					// Zero padding
					if (in_x < 0 || in_x >= input_width || in_y < 0 || in_y >= input_height) {
                    continue;
                }
                const int outer_base = Offset(input_shape, batch, in_y, in_x, group * filter_input_depth);
                const int8_t* act_base_ptr = input_data + outer_base;
                for (int in_channel = 0; in_channel < filter_input_depth; in_channel += 32) {
                    const int8_t* base = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                    int32_t val132 = *((const int32_t*)base);
                    int32_t val232 = *((const int32_t*)(base + 4));
                    // const int8_t* activations = act_base_ptr + in_channel;
                    // const uint32_t* act_words = (const uint32_t*)activations;
                    const uint32_t* act_words = (const uint32_t*)(act_base_ptr + in_channel);

                    cfu_op0(CFU_FUNCT7_PUSH_WEIGHTS, val132, val232);
                    cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[0], act_words[1]);
                    cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2], act_words[3]);
                    cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[4], act_words[5]);
                    cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[6], act_words[7]);
                }
              }
            }
            int32_t acc = cfu_op0(CFU_FUNCT7_MAC_READ, 0, 0);
            if (bias_data) {
              acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                static_cast<int8_t>(acc);
          }
        }
      }
    }
  return;
}


void NormalConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;


  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  int pcount=0;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
    if (bias_data) {
        TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    const int groups = input_depth / filter_input_depth;
    TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
    const int filters_per_group = output_depth / groups;
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    static bool print_once = true;


    int32_t start_cycle = 0;
    int32_t end_cycle = 0;
    int32_t cycle_count = 0;
    int32_t idx_count = 0;


    printf("===CONV2D_MARKER_START===\n");
    printf("CONV2D_CALL_INDEX:%d\n", conv2d_call_count);
    printf("stride_width:%d\n", stride_width);
    printf("stride_height:%d\n", stride_height);
    printf("dilation_width:%d\n", dilation_width_factor);
    printf("dilation_height:%d\n", dilation_height_factor);
    printf("pad_width:%d\n", pad_width);
    printf("pad_height:%d\n", pad_height);
    printf("output_offset:%d\n", output_offset);
    printf("output_activation_min:%d\n", output_activation_min);
    printf("output_activation_max:%d\n", output_activation_max);
    printf("input_shape:%d,%d,%d,%d\n", input_shape.Dims(0), input_shape.Dims(1), input_shape.Dims(2), input_shape.Dims(3));
    printf("filter_shape:%d,%d,%d,%d\n", filter_shape.Dims(0), filter_shape.Dims(1), filter_shape.Dims(2), filter_shape.Dims(3));
    printf("output_shape:%d,%d,%d,%d\n", output_shape.Dims(0), output_shape.Dims(1), output_shape.Dims(2), output_shape.Dims(3));
    printf("===CONV2D_MARKER_END===\n");


    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    int32_t acc = 0;

            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) &&
                    (in_y >= 0) && (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }
                for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x,
                      in_channel + group * filter_input_depth)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, out_channel, filter_y, filter_x, in_channel)];
                  acc += filter_val * (input_val + input_offset);
                  }
                }
              }
            if (bias_data) {
              acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                static_cast<int8_t>(acc);
          }
        }
      }
    }
/*
    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    int32_t acc = 0;
                    // FLATTENED LOOP STARTS HERE
                    const int total_kernel_size = filter_height * filter_width * filter_input_depth;
                    for (int flat_idx = 0; flat_idx < total_kernel_size; ++flat_idx) {
                        int in_channel = flat_idx % filter_input_depth;
                        int filter_x = (flat_idx / filter_input_depth) % filter_width;
                        int filter_y = (flat_idx / filter_input_depth) / filter_width;

                        int in_y = in_y_origin + dilation_height_factor * filter_y;
                        int in_x = in_x_origin + dilation_width_factor * filter_x;

                        bool is_point_inside_image =
                            (in_x >= 0) && (in_x < input_width) &&
                            (in_y >= 0) && (in_y < input_height);

                        if (!is_point_inside_image) {
                            continue;
                        }
                        int32_t input_val = input_data[Offset(
                            input_shape, batch, in_y, in_x,
                            in_channel + group * filter_input_depth)];
                        int32_t filter_val = filter_data[Offset(
                            filter_shape, out_channel, filter_y, filter_x, in_channel)];
                        acc += filter_val * (input_val + input_offset);
                    }
                    // FLATTENED LOOP ENDS HERE

                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(
                        acc, output_multiplier[out_channel], output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                        static_cast<int8_t>(acc);
                }
            }
        }
    }
    */
}

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
#ifdef CONV_ACCELERATE
    // bool is_acc = is_accelerated_conv_layer(conv2d_call_count);
    // if (is_acc)
    // if (conv2d_call_count == 7)
    if (conv2d_call_count == 4 || conv2d_call_count == 6 || conv2d_call_count == 7 || conv2d_call_count == 8)
    {
    AccelConvPerChannel(
        params, output_multiplier, output_shift,
        input_shape, input_data, filter_shape, filter_data,
        bias_shape, bias_data, output_shape, output_data
    );
    }
    else if (conv2d_call_count == 1 || conv2d_call_count == 2 || conv2d_call_count == 3 || conv2d_call_count == 5)
    {
    AccelConvPerChannel_16(
        params, output_multiplier, output_shift,
        input_shape, input_data, filter_shape, filter_data,
        bias_shape, bias_data, output_shape, output_data
    );
    }
    else{
    NormalConvPerChannel(
        params, output_multiplier, output_shift,
        input_shape, input_data, filter_shape, filter_data,
        bias_shape, bias_data, output_shape, output_data
    );
    }
#else
    NormalConvPerChannel(
        params, output_multiplier, output_shift,
        input_shape, input_data, filter_shape, filter_data,
        bias_shape, bias_data, output_shape, output_data
    );
  // Get parameters.
#endif
    conv2d_call_count++;
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;


  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
        //


                    /*
                  if (print_once == true){
                      printf("FV 1 %d %d \n ", filter_val1, filter_val2);
                    // Inspect active_clusters[0] from CFU
                    print_bits(val132);
                    printf("\n");
                    print_bits(val232);
                    printf("\n");
                    for (int i = 0; i < 32; i++) {
                      int val = cfu_op0_hw(CFU_FUNCT7_DEBUG_DUMP, i, 0);
                          printf("active_clusters[%d] = %d\n", i, val);
                    }
                    for (int in_channel = 0; in_channel < filter_input_depth; in_channel++) {
                            int32_t filter_val = filter_data[Offset(
                                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                            printf("Raw Weight: in_channel=%d, value=%d\n", in_channel, filter_val);
                    }


                    // Inspect activations
                    printf("[DEBUG] Activations:\n  ");
                    for (int i = 0; i < 32; i++) {
                    printf("%4d ", activations[i]);
                    if ((i+1) % 8 == 0) printf("\n  ");
                    }
                    printf("\n");
                    print_once = false;
                  }
                  */

/*
                  if (print_once == true){
                    printf("\n");
                    print_bits(val132);
                    printf("\n");
                    printf("\n");
                    print_bits(val232);
                    printf("\n");
                    printf("in_channel %d %d \n", in_channel, filter_x);
                    for (int i = 0; i < 32; i++) {
                      int val = cfu_op0_hw(CFU_FUNCT7_DEBUG_DUMP, i, 0);
                          printf("active_clusters[%d] = %d\n", i, val);
                    }
                    // Inspect activations
                    printf("[DEBUG] Activations:\n  ");
                    for (int i = 0; i < 32; i++) {
                    printf("%4d ", activations[i]);
                    if ((i+1) % 8 == 0) printf("\n  ");
                    }
                    int32_t acc = cfu_op0(CFU_FUNCT7_MAC_READ_NO_RESET, 0, 0);
                    printf(" ACC DEBUG %d \n", acc);
                  }
                  */
