#include "direct_convolution.h"
#include <arm_neon.h>

typedef float real32_t;
void loop_order_n1(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int l = 0; l < out_height; ++l) {
    for (int n = 0; n < kernel_height; ++n) {
      for (int m = 0; m < kernel_width; ++m) {
        for (int i = 0; i < input_channels; ++i) {
          for (int k = 0; k < out_width; ++k) {
            for (int j = 0; j < output_channels; ++j) {
              result[l * out_width * output_channels + k * output_channels + j] +=
                kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j] *
                image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
            }
          }
        }
      }
    }
  }
}


void loop_order_n2(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int n = 0; n < kernel_height; ++n) {
    for (int m = 0; m < kernel_width; ++m) {
      for (int i = 0; i < input_channels; ++i) {
        for (int l = 0; l < out_height; ++l) {
          for (int k = 0; k < out_width; ++k) {
            for (int j = 0; j < output_channels; ++j) {
              result[l * out_width * output_channels + k * output_channels + j] +=
                kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) +
                       i * output_channels + j] *
                image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
            }
          }
        }
      }
    }
  }
}


void loop_order_n3(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int l = 0; l < out_height; ++l) {
    for (int n = 0; n < kernel_height; ++n) {
      for (int m = 0; m < kernel_width; ++m) {
        for (int k = 0; k < out_width; ++k) {
          for (int i = 0; i < input_channels; ++i) {
            for (int j = 0; j < output_channels; ++j) {
              result[l * out_width * output_channels + k * output_channels + j] +=
                kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j] *
                image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
            }
          }
        }
      }
    }
  }
}

void loop_order_n1_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;

  for (int l = 0; l < out_height; ++l) {
    for (int n = 0; n < kernel_height; ++n) {
      for (int m = 0; m < kernel_width; ++m) {
        for (int i = 0; i < input_channels; ++i) {
          for (int k = 0; k < out_width; ++k) {
            image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];

            for (int j = 0; j < output_channels; j += 8) {
              kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
              image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
              kernel_data = vld1q_f32_x2(kernel_ptr);
              result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);
              result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
              result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
              vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
            }
          }
        }
      }
    }
  }
}


void loop_order_n2_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;

  for (int n = 0; n < kernel_height; ++n) {
    for (int m = 0; m < kernel_width; ++m) {
      for (int i = 0; i < input_channels; ++i) {
        for (int l = 0; l < out_height; ++l) {
          for (int k = 0; k < out_width; ++k) {
            image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];

            for (int j = 0; j < output_channels; j += 8) {
              kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
              image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
              kernel_data = vld1q_f32_x2(kernel_ptr);
              result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);

              result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
              result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
              vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
            }
          }
        }
      }
    }
  }
}


void loop_order_n3_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;
  for (int l = 0; l < out_height; ++l) {
    for (int n = 0; n < kernel_height; ++n) {
      for (int m = 0; m < kernel_width; ++m) {
        for (int k = 0; k < out_width; ++k) {
          for (int i = 0; i < input_channels; ++i) {
            image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];

            for (int j = 0; j < output_channels; j += 8) {
              kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
              image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
              kernel_data = vld1q_f32_x2(kernel_ptr);
              result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);

              result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
              result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
              vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
            }
          }
        }
      }
    }
  }
}



void loop_order_n1_tiled(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int ii = 0; ii < input_channels; ii += tile_size_input_channels) {
    for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int i = ii; i < ii + tile_size_input_channels && i < input_channels; ++i) {
              for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
                for (int j = 0; j < output_channels; ++j) {
                  result[l * out_width * output_channels + k * output_channels + j] +=
                    kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j] *
                    image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n2_tiled(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_out_width, int tile_size_out_height) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
    for (int ll = 0; ll < out_height; ll += tile_size_out_height) {
      for (int n = 0; n < kernel_height; ++n) {
        for (int m = 0; m < kernel_width; ++m) {
          for (int i = 0; i < input_channels; ++i) {
            for (int l = ll; l < ll + tile_size_out_height && l < out_height; ++l) {
              for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
                for (int j = 0; j < output_channels; ++j) {
                  result[l * out_width * output_channels + k * output_channels + j] +=
                    kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) +
                           i * output_channels + j] *
                    image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n3_tiled(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  for (int ii= 0; ii < input_channels; ii += tile_size_input_channels) {
    for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
              for (int i = ii; i < ii + tile_size_input_channels && i < input_channels; ++i) {
                for (int j = 0; j < output_channels; ++j) {
                  result[l * out_width * output_channels + k * output_channels + j] +=
                    kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j] *
                    image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                }
              }
            }
          }
        }
      }
    }
  }
}




void loop_order_n1_tiled_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;

  for (int ii = 0; ii < input_channels; ii += tile_size_input_channels) {
    for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int i = ii; i < ii + tile_size_input_channels && i < input_channels; ++i) {
              for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
                image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                for (int j = 0; j < output_channels; j += 8) {
                  kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
                  image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
                  kernel_data = vld1q_f32_x2(kernel_ptr);
                  result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);
                  result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
                  result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
                  vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n2_tiled_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_out_width, int tile_size_out_height) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;

  for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
    for (int ll = 0; ll < out_height; ll += tile_size_out_height) {
      for (int n = 0; n < kernel_height; ++n) {
        for (int m = 0; m < kernel_width; ++m) {
          for (int i = 0; i < input_channels; ++i) {
            for (int l = ll; l < ll + tile_size_out_height && l < out_height; ++l) {
              for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
                image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                for (int j = 0; j < output_channels; j += 8) {
                  kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
                  image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
                  kernel_data = vld1q_f32_x2(kernel_ptr);
                  result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);
                  result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
                  result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
                  vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n3_tiled_with_simd(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  float32x4x2_t result_buffer;
  float32x4x2_t image_data;
  float32x4x2_t kernel_data;
  const real32_t *kernel_ptr;
  const real32_t *image_ptr;

  for (int ii = 0; ii < input_channels; ii += tile_size_input_channels) {
    for (int kk = 0; kk < out_width; kk += tile_size_out_width) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int k = kk; k < kk + tile_size_out_width && k < out_width; ++k) {
              for (int i = ii; i < ii + tile_size_input_channels && i < input_channels; ++i) {
                image_ptr = &image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels + i];
                for (int j = 0; j < output_channels; j += 8) {
                  kernel_ptr = &kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) + i * output_channels + j];
                  image_data = {vdupq_n_f32(image_ptr[0]), vdupq_n_f32(image_ptr[0])};
                  kernel_data = vld1q_f32_x2(kernel_ptr);
                  result_buffer = vld1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j]);
                  result_buffer.val[0] = vmlaq_f32(result_buffer.val[0], image_data.val[0], kernel_data.val[0]);
                  result_buffer.val[1] = vmlaq_f32(result_buffer.val[1], image_data.val[1], kernel_data.val[1]);
                  vst1q_f32_x2(&result[l * out_width * output_channels + k * output_channels + j], result_buffer);
                }
              }
            }
          }
        }
      }
    }
  }
}
