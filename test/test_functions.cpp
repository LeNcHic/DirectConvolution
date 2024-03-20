#include "test_functions.h"
typedef float real32_t;

void loop_order_n1_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int l = 0; l < out_height; l += tile_size) {
    for (int n = 0; n < kernel_height; n += tile_size) {
      for (int m = 0; m < kernel_width; m += tile_size) {
        for (int i = 0; i < input_channels; i += tile_size) {
          for (int k = 0; k < out_width; k += tile_size) {
            for (int j = 0; j < output_channels; j += tile_size) {
              //Tiling loops
              for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                  for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                    for (int it = 0; it < tile_size && (i + it) < input_channels; ++it) {
                      for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                        for (int jt = 0; jt < tile_size && (j + jt) < output_channels; ++jt) {
                          result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + (j + jt)] +=
                            kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                                   (m + mt) * (input_channels * output_channels) + (i + it) * output_channels + (j + jt)] *
                            image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                                  ((m + mt) * stride_width + (k + kt)) * input_channels + (i + it)];
                        }

                      }
                    }

                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n2_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int n = 0; n < kernel_height; n += tile_size) {
    for (int m = 0; m < kernel_width; m += tile_size) {
      for (int i = 0; i < input_channels; i += tile_size) {
        for (int l = 0; l < out_height; l += tile_size) {
          for (int k = 0; k < out_width; k += tile_size) {
            for (int j = 0; j < output_channels; j += tile_size) {
              //Tiling loops
              for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                  for (int it = 0; it < tile_size && (i + it) < input_channels; ++it) {
                    for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                      for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                        for (int jt = 0; jt < tile_size && (j + jt) < output_channels; ++jt) {
                          result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + (j + jt)] +=
                            kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                                   (m + mt) * (input_channels * output_channels) + (i + it) * output_channels + (j + jt)] *
                            image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                                  ((m + mt) * stride_width + (k + kt)) * input_channels + (i + it)];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


void loop_order_n3_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int l = 0; l < out_height; l += tile_size) {
    for (int n = 0; n < kernel_height; n += tile_size) {
      for (int m = 0; m < kernel_width; m += tile_size) {
        for (int k = 0; k < out_width; k += tile_size) {
          for (int i = 0; i < input_channels; i += tile_size) {
            for (int j = 0; j < output_channels; j += tile_size) {
              //Tiling loops
              for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                  for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                    for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                      for (int it = 0; it < tile_size && (i + it) < input_channels; ++it) {
                        for (int jt = 0; jt < tile_size && (j + jt) < output_channels; ++jt) {
                          result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + (j + jt)] +=
                            kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                                   (m + mt) * (input_channels * output_channels) + (i + it) * output_channels + (j + jt)] *
                            image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                                  ((m + mt) * stride_width + (k + kt)) * input_channels + (i + it)];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}



void loop_order_n1_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int ii = 0; ii < input_channels; ii += tile_size) {
    for (int kk = 0; kk < out_width; kk += tile_size) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int i = ii; i < ii + tile_size && i < input_channels; ++i) {
              for (int k = kk; k < kk + tile_size && k < out_width; ++k) {
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



void loop_order_n2_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int kk = 0; kk < out_width; kk += tile_size) {
    for (int ll = 0; ll < out_height; ll += tile_size) {
      for (int n = 0; n < kernel_height; ++n) {
        for (int m = 0; m < kernel_width; ++m) {
          for (int i = 0; i < input_channels; ++i) {
            for (int l = ll; l < ll + tile_size && l < out_height; ++l) {
              for (int k = kk; k < kk + tile_size && k < out_width; ++k) {
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


void loop_order_n3_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size) {

  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;

  for (int ii = 0; ii < input_channels; ii += tile_size) {
    for (int kk = 0; kk < out_width; kk += tile_size) {
      for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
          for (int m = 0; m < kernel_width; ++m) {
            for (int k = kk; k < kk + tile_size && k < out_width; ++k) {
              for (int i = 0; i < ii + tile_size && i < input_channels; ++i) {
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
