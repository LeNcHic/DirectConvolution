typedef float real32_t;

void loop_order_n1_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);

void loop_order_n2_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);

void loop_order_n3_tiled_every(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);

void loop_order_n1_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);

void loop_order_n2_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);

void loop_order_n3_tiled_changed_order(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int tile_size);
