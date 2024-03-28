#ifndef ELBRUS_CONVOLUTION_DIRECT_CONVOLUTION_H
#define ELBRUS_CONVOLUTION_DIRECT_CONVOLUTION_H
typedef float real32_t;

void loop_order_n1(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);
void loop_order_n2(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);
void loop_order_n3(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);

void loop_order_n1_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);
void loop_order_n2_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);
void loop_order_n3_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width);

void loop_order_n1_tiled(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width);
void loop_order_n2_tiled(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_width, int tile_size_out_height);
void loop_order_n3_tiled(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width);

void loop_order_n1_tiled_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width);
void loop_order_n2_tiled_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_width, int tile_size_out_height);
void loop_order_n3_tiled_with_simd(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels, int tile_size_out_width);

void loop_order_n1_tiled_channels(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels);
void loop_order_n1_tiled_width(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_width);
void loop_order_n2_tiled_width(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_width);
void loop_order_n2_tiled_height(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_height);
void loop_order_n3_tiled_channels(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_input_channels);
void loop_order_n3_tiled_width(real32_t *result, const real32_t *image, const real32_t *kernel, int input_height, int input_width, int input_channels, int stride_height, int stride_width, int output_channels, int kernel_height, int kernel_width, int tile_size_out_width);


#endif //ELBRUS_CONVOLUTION_DIRECT_CONVOLUTION_H
