typedef float real32_t;

void loop_order_n1(
        real32_t *result,
        const real32_t *image, const real32_t *kernel,
        int input_height, int input_width, int input_channels,
        int stride_height, int stride_width, int output_channels,
        int kernel_height, int kernel_width) {

    const int out_height = (input_height - kernel_height + stride_height) /stride_height;
    const int out_width = (input_width - kernel_width + stride_width) /stride_width;

    for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
            for (int m = 0; m < kernel_width; ++m) {
                for (int i = 0; i < input_channels; ++i) {
                    for (int k = 0; k < out_width; ++k) {
                        for (int j = 0; j < output_channels; ++j) {
                            result[(l * kernel_height + n) * kernel_width + m] +=
                                    kernel[(i * stride_height + l) * output_channels * kernel_width + (j * stride_width + m) * kernel_height + n] *
                                    image[(i * input_height + k * stride_height + l) * input_width * output_channels + (j * stride_width + m)];
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

    const int out_height = (input_height - kernel_height + stride_height) /stride_height;
    const int out_width = (input_width - kernel_width + stride_width) /stride_width;

    for (int n = 0; n < kernel_height; ++n) {
        for (int m = 0; m < kernel_width; ++m) {
            for (int i = 0; i < input_channels; ++i) {
                for (int l = 0; l < out_height; ++l) {
                    for (int k = 0; k < out_width; ++k) {
                        for (int j = 0; j < output_channels; ++j) {
                            result[(l * kernel_height + n) * kernel_width + m] +=
                                    kernel[(i * stride_height + l) * output_channels * kernel_width + (j * stride_width + m) * kernel_height + n] *
                                    image[(i * input_height + k * stride_height + l) * input_width * output_channels + (j * stride_width + m)];
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

    const int out_height = (input_height - kernel_height + stride_height) /stride_height;
    const int out_width = (input_width - kernel_width + stride_width) /stride_width;

    for (int l = 0; l < out_height; ++l) {
        for (int n = 0; n < kernel_height; ++n) {
            for (int m = 0; m < kernel_width; ++m) {
                for (int k = 0; k < out_width; ++k) {
                    for (int i = 0; i < input_channels; ++i) {
                        for (int j = 0; j < output_channels; ++j) {
                            result[(l * kernel_height + n) * kernel_width + m] +=
                                    kernel[(i * stride_height + l) * output_channels * kernel_width + (j * stride_width + m) * kernel_height + n] *
                                    image[(i * input_height + k * stride_height + l) * input_width * output_channels + (j * stride_width + m)];
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
        int kernel_height, int kernel_width, int tile_size) {

    const int out_height = (input_height - kernel_height + stride_height) / stride_height;
    const int out_width = (input_width - kernel_width + stride_width) / stride_width;

    for (int l = 0; l < out_height; l += tile_size) {
        for (int n = 0; n < kernel_height; n += tile_size) {
            for (int m = 0; m < kernel_width; m += tile_size) {
                for (int i = 0; i < input_channels; ++i) {
                    for (int k = 0; k < out_width; k += tile_size) {
                        for (int j = 0; j < output_channels; ++j) {
                            //Tiling loops
                            for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                                for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                                    for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                                        for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                                            result[((l + lt) * kernel_height + (n + nt)) * kernel_width + (m + mt)] +=
                                                    kernel[(i * stride_height + (l + lt)) * output_channels * kernel_width +
                                                           ((j * stride_width + (m + mt)) * kernel_height + (n + nt))] *
                                                    image[(i * input_height + (k + kt) * stride_height + (l + lt)) * input_width * output_channels +
                                                          ((j * stride_width + (m + mt)) * input_channels)];
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


void loop_order_n2_tiled(
        real32_t *result,
        const real32_t *image, const real32_t *kernel,
        int input_height, int input_width, int input_channels,
        int stride_height, int stride_width, int output_channels,
        int kernel_height, int kernel_width, int tile_size) {

    const int out_height = (input_height - kernel_height + stride_height) / stride_height;
    const int out_width = (input_width - kernel_width + stride_width) / stride_width;

    for (int n = 0; n < kernel_height; n += tile_size) {
        for (int m = 0; m < kernel_width; m += tile_size) {
            for (int i = 0; i < input_channels; ++i) {
                for (int l = 0; l < out_height; l += tile_size) {
                    for (int k = 0; k < out_width; k += tile_size) {
                        for (int j = 0; j < output_channels; ++j) {
                            //Tiling loops
                            for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                                for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                                    for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                                        for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                                            result[((l + lt) * kernel_height + (n + nt)) * kernel_width + (m + mt)] +=
                                                    kernel[(i * stride_height + (l + lt)) * output_channels * kernel_width +
                                                           ((j * stride_width + (m + mt)) * kernel_height + (n + nt))] *
                                                    image[(i * input_height + (k + kt) * stride_height + (l + lt)) * input_width * output_channels +
                                                          ((j * stride_width + (m + mt)) * input_channels)];
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


void loop_order_n3_tiled(
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
                    for (int i = 0; i < input_channels; ++i) {
                        for (int j = 0; j < output_channels; ++j) {
                            //Tiling loops
                            for (int lt = 0; lt < tile_size && (l + lt) < out_height; ++lt) {
                                for (int nt = 0; nt < tile_size && (n + nt) < kernel_height; ++nt) {
                                    for (int mt = 0; mt < tile_size && (m + mt) < kernel_width; ++mt) {
                                        for (int kt = 0; kt < tile_size && (k + kt) < out_width; ++kt) {
                                            result[((l + lt) * kernel_height + (n + nt)) * kernel_width + (m + mt)] +=
                                                    kernel[(i * stride_height + (l + lt)) * output_channels * kernel_width +
                                                           ((j * stride_width + (m + mt)) * kernel_height + (n + nt))] *
                                                    image[((i) * input_height + (k + kt) * stride_height + (l + lt)) * input_width * output_channels +
                                                          ((j * stride_width + (m + mt)) * input_channels)];
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
