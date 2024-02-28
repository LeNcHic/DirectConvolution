#include <chrono>
#include <cstring>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <iomanip>
//#include "directConvolution.cpp"

typedef float real32_t;

# ifndef MUSTINLINE
#   ifdef _MSC_VER
#     define MUSTINLINE inline __forceinline
#   else
#     define MUSTINLINE inline __attribute__((always_inline))
#   endif
# endif

template<typename T>
class AlignedBuffer {
public:
    AlignedBuffer() {
      p_src = nullptr;
    }

    void set_size(int v_size) {
      free(p_src);
      size = v_size;
      //выделение памяти с выравниванием на границу 32 байта
      posix_memalign(reinterpret_cast<void **>(&p_src), 32, size * sizeof(T));
    };

    T *data() const {

      return p_src;
    }

    T &operator[](int i) {
      return *(p_src + i);
    }

    //rule of 5
    AlignedBuffer(const AlignedBuffer<T> &other) {
      p_src = nullptr;
      set_size(other.size);
      std::copy(other.p_src, other.p_src + size, p_src);
    }

    AlignedBuffer(AlignedBuffer<T> &&other) noexcept: size(other.size), p_src(other.p_src) {
      other.size = 0;
      other.p_src = nullptr;
    }

    AlignedBuffer &operator=(const AlignedBuffer<T> &other) {
      if (&other != this) {
        set_size(other.size);
        std::copy(other.p_src, other.p_src + size, p_src);
      }
      return *this;
    }

    AlignedBuffer &operator=(AlignedBuffer<T> &&other) noexcept {
      if (&other != this) {
        free(p_src);
        size = other.size;
        p_src = other.p_src;
        other.size = 0;
        other.p_src = nullptr;
      }
      return *this;
    }

    ~AlignedBuffer() {
      free(p_src);
    }

private:
    int size = 0;
    T *p_src;
};

struct MinSize {
    int width, height;   // width and height values of the size
    MinSize(int _width = 0, int _height = 0) : width(_width), height(_height) {}

};


template<typename T>
bool
img_equal(const T *img1, int stride1, const T *img2, int stride2, int height, int width, int channels, T eps = T()) {
  const T *ptr1 = img1;
  const T *ptr2 = img2;
  for (int i = 0; i < height; ++i, ptr1 += stride1, ptr2 += stride2) {
    for (int j = 0; j < width * channels; ++j) {
      T delta = ptr1[j] - ptr2[j];
      if (std::abs(delta) > eps) {
        return false;
      }
    }
  }
  return true;
}

template<>
bool img_equal<real32_t>(const real32_t *img1, int stride1, const real32_t *img2, int stride2, int height, int width,
                         int channels, real32_t eps) {
  const real32_t *ptr1 = img1;
  const real32_t *ptr2 = img2;
  for (int i = 0; i < height; ++i, ptr1 += stride1, ptr2 += stride2) {
    for (int j = 0; j < width * channels; ++j) {
      real32_t etta = (ptr1[j] - ptr2[j]) / (std::abs(ptr1[j]) + 0.1f);
      if (std::abs(etta) > eps) {
        return false;
      }
    }
  }
  return true;
}


template<typename T>
MUSTINLINE void im2col(
  T *im2col_buffer, const T *image, int image_stride, int out_height, int out_width, int n_channels,
  int kernel_height, int kernel_width, int stride_height, int stride_width) {

  real32_t *p_dst = im2col_buffer;
  for (int j = 0; j < out_height; ++j) {
    for (int i = 0; i < out_width; ++i) {
      //copy row by row
      int srcRowInitNumber = j * stride_height;
      int srcColInitNumber = i * stride_width;
      for (int r = 0; r < kernel_height; ++r) {
        const real32_t *p_start = image + (srcRowInitNumber + r) * image_stride + srcColInitNumber * n_channels;
        std::copy(p_start, p_start + kernel_width * n_channels, p_dst);
        p_dst += kernel_width * n_channels;
      }
    }
  }
}

template<typename T>
MUSTINLINE int im2col(
  T *im2col_buffer, const T *image, int image_stride, int out_height, int out_width, int n_channels,
  int kernel_height, int kernel_width, int stride_height, int stride_width, int &i, int &j, int max_rows) {

  int cnt = max_rows;
  real32_t *p_dst = im2col_buffer;
  for (; j < out_height; ++j) {
    for (; i < out_width && cnt > 0; ++i) {
      //copy row by row
      int srcRowInitNumber = j * stride_height;
      int srcColInitNumber = i * stride_width;
      for (int r = 0; r < kernel_height; ++r) {
        const real32_t *p_start = image + (srcRowInitNumber + r) * image_stride + srcColInitNumber * n_channels;
        std::copy(p_start, p_start + kernel_width * n_channels, p_dst);
        p_dst += kernel_width * n_channels;
      }
      --cnt;
    }
    if (cnt > 0) {
      i = 0;
    } else {
      break;
    }
  }
  return max_rows - cnt;
}

void im2col_convolution(
  real32_t *result, real32_t *im2col_buffer,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width) {
  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  const int depth = input_channels * kernel_width * kernel_height;
  const int width = output_channels;
  const int height = out_height * out_width;
  im2col(im2col_buffer, image, input_width * input_channels, out_height, out_width, input_channels,
         kernel_height, kernel_width, stride_height, stride_width);
  Eigen::Map<const Eigen::MatrixXf> right(im2col_buffer, depth, height);
  Eigen::Map<const Eigen::MatrixXf> left(kernel, width, depth);
  Eigen::Map<Eigen::MatrixXf> res(result, width, height);
  res.noalias() = left * right;
}

void block_im2col_convolution(
  real32_t *result, real32_t *im2col_buffer,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int stride_height, int stride_width, int output_channels,
  int kernel_height, int kernel_width, int param) {
  const int out_height = (input_height - kernel_height + stride_height) / stride_height;
  const int out_width = (input_width - kernel_width + stride_width) / stride_width;
  const int depth = input_channels * kernel_width * kernel_height;
  const int width = output_channels;

  Eigen::Map<const Eigen::MatrixXf> left(kernel, width, depth);
  int i = 0, j = 0;

  real32_t *p_dst = result;
  while (j < out_height) {
    const int height = im2col(im2col_buffer, image, input_width * input_channels, out_height, out_width, input_channels,
                              kernel_height, kernel_width, stride_height, stride_width, i, j, param);
    Eigen::Map<const Eigen::MatrixXf> right(im2col_buffer, depth, height);
    Eigen::Map<Eigen::MatrixXf> res(p_dst, width, height);
    res.noalias() = left * right;
    p_dst += height * width;
  }
}

void aa_convolution(
  real32_t *result,
  const real32_t *image, const real32_t *kernel,
  int input_height, int input_width, int input_channels,
  int /*stride_height*/, int /*stride_width*/, int output_channels,
  int kernel_height, int kernel_width) {
  const int height = input_height * input_width;
  const int width = output_channels;
  const int depth = input_channels;
  int aa_offset = ((kernel_width + kernel_height * input_width) * output_channels + 31) / 32 * 32;
  int aa_kernel_size = input_channels * output_channels;
  const real32_t *ker_ptr = kernel;
  std::fill(result, result + height * width + aa_offset, 0.f);
  result += aa_offset;
  for (int kh = 0; kh < kernel_height; ++kh) {
    for (int kw = 0; kw < kernel_width; ++kw) {
      int result_shift = (kw + kh * input_width) * output_channels;
      Eigen::Map<const Eigen::MatrixXf> right(image, depth, height);
      Eigen::Map<const Eigen::MatrixXf> left(ker_ptr, width, depth);
      Eigen::Map<Eigen::MatrixXf> res(result - result_shift, width, height);
      res.noalias() += left * right;
      ker_ptr += aa_kernel_size;
    }
  }
}

void direct_convolution() {

}


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
                kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) +
                       i * output_channels + j] *
                image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels +
                      i];
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
                kernel[n * (kernel_width * input_channels * output_channels) + m * (input_channels * output_channels) +
                       i * output_channels + j] *
                image[(n * stride_height + l) * input_width * input_channels + (m * stride_width + k) * input_channels +
                      i];
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
                      result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + j] +=
                        kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                               (m + mt) * (input_channels * output_channels) + i * output_channels + j] *
                        image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                              ((m + mt) * stride_width + (k + kt)) * input_channels + i];
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
                      result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + j] +=
                        kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                               (m + mt) * (input_channels * output_channels) + i * output_channels + j] *
                        image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                              ((m + mt) * stride_width + (k + kt)) * input_channels + i];
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
                      result[(l + lt) * out_width * output_channels + (k + kt) * output_channels + j] +=
                        kernel[(n + nt) * (kernel_width * input_channels * output_channels) +
                               (m + mt) * (input_channels * output_channels) + i * output_channels + j] *
                        image[((n + nt) * stride_height + (l + lt)) * input_width * input_channels +
                              ((m + mt) * stride_width + (k + kt)) * input_channels + i];
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


enum ConvType {
    CONV_TYPE_NONE = 0,
    CONV_TYPE_IM2COL_F32 = 1,
    CONV_TYPE_IM2COL_PARTED_F32 = 2,
    CONV_TYPE_ANDERSON_F32 = 3,
    CONV_TYPE_LOOP1 = 4,
    CONV_TYPE_LOOP2 = 5,
    CONV_TYPE_LOOP3 = 6,
    CONV_TYPE_TILING_LOOP1 = 7,
    CONV_TYPE_TILING_LOOP2 = 8,
    CONV_TYPE_TILING_LOOP3 = 9,


};

struct ConvWrapper {
    ConvType ct = CONV_TYPE_NONE;
    MinSize in_size = MinSize(0, 0);
    int in_channels = 0;
    MinSize filter_size = MinSize(0, 0);
    int filters_count = 0;
    int groups = 1;
    MinSize stride = MinSize(1, 1);
    int parted_rows = 1;
    MinSize out_size;
    int out_channels;

    AlignedBuffer<real32_t> buffer;
    AlignedBuffer<real32_t> matrix;
    AlignedBuffer<real32_t> input_image;
    AlignedBuffer<real32_t> result;


    // не менял
    void set() {
      out_size = MinSize(
        (in_size.width - filter_size.width + stride.width) / stride.width,
        (in_size.height - filter_size.height + stride.height) / stride.height
      );
      out_channels = filters_count * groups;
      input_image.set_size(in_size.height * in_size.width * in_channels);
      int res_size = out_size.height * out_size.width * out_channels;
      if (ct == CONV_TYPE_ANDERSON_F32) {
        int aa_offset = ((filter_size.height * in_size.width + filter_size.width) * out_channels + 31) / 32 * 32;
        res_size = in_size.height * in_size.width * out_channels + aa_offset;
      }
      result.set_size(res_size);
      matrix.set_size(out_channels * in_channels / groups * filter_size.height * filter_size.width);

      const int height = out_size.height * out_size.width;
      const int depth = in_channels * filter_size.height * filter_size.width;

      if (ct == CONV_TYPE_IM2COL_F32) {
        buffer.set_size(height * depth);
      } else if (ct == CONV_TYPE_IM2COL_PARTED_F32) {
        buffer.set_size(parted_rows * depth);
      }
    }

    void init();

    void run() {
      switch (ct) {
        case CONV_TYPE_NONE:
          break;
        case CONV_TYPE_IM2COL_F32:
          im2col_convolution(result.data(), buffer.data(), input_image.data(), matrix.data(),
                             in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                             filter_size.height, filter_size.width);
          break;
        case CONV_TYPE_IM2COL_PARTED_F32:
          block_im2col_convolution(result.data(), buffer.data(), input_image.data(), matrix.data(),
                                   in_size.height, in_size.width, in_channels, stride.height, stride.width,
                                   out_channels,
                                   filter_size.height, filter_size.width, parted_rows);
          break;
        case CONV_TYPE_ANDERSON_F32:
          aa_convolution(result.data(), input_image.data(), matrix.data(),
                         in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                         filter_size.height, filter_size.width);
          break;
        case CONV_TYPE_LOOP1:
          loop_order_n1(result.data(), input_image.data(), matrix.data(),
                        in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                        filter_size.height, filter_size.width);
          break;
        case CONV_TYPE_LOOP2:
          loop_order_n2(result.data(), input_image.data(), matrix.data(),
                        in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                        filter_size.height, filter_size.width);
          break;
        case CONV_TYPE_LOOP3:
          loop_order_n3(result.data(), input_image.data(), matrix.data(),
                        in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                        filter_size.height, filter_size.width);
          break;
        case CONV_TYPE_TILING_LOOP1:
          loop_order_n1_tiled(result.data(), input_image.data(), matrix.data(),
                              in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                              filter_size.height, filter_size.width, 128);
          break;
        case CONV_TYPE_TILING_LOOP2:
          loop_order_n2_tiled(result.data(), input_image.data(), matrix.data(),
                              in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                              filter_size.height, filter_size.width, 128);
          break;
        case CONV_TYPE_TILING_LOOP3:
          loop_order_n3_tiled(result.data(), input_image.data(), matrix.data(),
                              in_size.height, in_size.width, in_channels, stride.height, stride.width, out_channels,
                              filter_size.height, filter_size.width, 128);

      }
    }

    void measure(double &time, double &error, double &total_time, int &n_runs, double one_test_time = 0.1) {
      static const int AVERAGE_REP = 5;
      set();
      init();
      auto test_start = std::chrono::high_resolution_clock::now();
      run();
      auto test_end = std::chrono::high_resolution_clock::now();
      double test_time = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(test_end - test_start).count();
      int n_rep = std::max(1, static_cast<int>(one_test_time / test_time));
      std::array<double, AVERAGE_REP> measurements{};
      total_time = test_time;
      std::generate(measurements.begin(), measurements.end(), [&]() {
          init();
          auto start = std::chrono::high_resolution_clock::now();
          for (int i = 0; i < n_rep; ++i) {
            run();
          }
          auto end = std::chrono::high_resolution_clock::now();
          double m_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
          total_time += 1e-6 * m_time;
          m_time *= 1e-3 / n_rep;
          return m_time;
      });
      std::sort(measurements.begin(), measurements.end());
      time = (measurements[(AVERAGE_REP - 1) / 2] + measurements[AVERAGE_REP / 2]) * 0.5;
      n_runs = 1 + AVERAGE_REP * n_rep;
      double t = 0, t2 = 0;
      for (auto &m: measurements) {
        t += m;
        t2 += m * m;
      }
      t /= AVERAGE_REP;
      t2 /= AVERAGE_REP;
      error = std::sqrt((t2 - t * t) / (AVERAGE_REP - 1));
    }
};


bool check_correctness(ConvWrapper im2col, ConvWrapper direct_convolution, int out_channels, int out_height, int out_width) {
  bool flag = true;
  for (int j = 0; j < out_channels; ++j) {
    for (int k = 0; k < out_width; ++k) {
      for (int l = 0; l < out_height; ++l) {

        if (im2col.result[l * out_width * out_channels + k * out_channels + j] != direct_convolution.result[l * out_width * out_channels + k * out_channels + j]) {
          flag = false;
          std::cout << "im2: " << im2col.result[l * out_width * out_channels + k * out_channels + j] << "    ";
          std::cout << "dc: " << direct_convolution.result[l * out_width * out_channels + k * out_channels + j] << std::endl;
        }
      }
    }
  }
  std::cout << flag;
  return flag;
}

void print_image(ConvWrapper a) {
  for (int j = 0; j < a.in_channels; ++j) {
    for (int k = 0; k < a.in_size.width; ++k) {
      for (int l = 0; l < a.in_size.height; ++l) {
        std::cout << a.input_image[l * a.in_size.width * a.in_channels + k * a.in_channels + j] << " ";
      }
    }
  }
  std::cout << std::endl;
}

// заполняем случайными значенияси изображение и матрицу
void ConvWrapper::init() {
  std::random_device r;
  std::minstd_rand gen(r());
  std::normal_distribution<real32_t> dist(0, 10);
  auto generator = [&]() { return dist(gen); };

  std::generate(input_image.data(), input_image.data() + in_size.height * in_size.width * in_channels, generator);
  std::generate(matrix.data(),
                matrix.data() + out_channels * in_channels / groups * filter_size.height * filter_size.width,
                generator);
}

MUSTINLINE void report(const ConvWrapper &cw, double time, double error, double total_time, int n_runs) {
  switch (cw.ct) {
    case CONV_TYPE_NONE:
      std::cout << "none:\t\t\t\t";
      break;
    case CONV_TYPE_IM2COL_F32:
      std::cout << "\tim2col_f32:\t\t";
      break;
    case CONV_TYPE_IM2COL_PARTED_F32:
      std::cout << "\tadaptive_f32 " << std::setw(4) << cw.parted_rows << ":\t";
      break;
    case CONV_TYPE_ANDERSON_F32:
      std::cout << "\taa_algo_f32:\t";
      break;
    case CONV_TYPE_LOOP1:
      std::cout << "\tloop_order_1\t";
      break;
    case CONV_TYPE_LOOP2:
      std::cout << "\tloop_order_2\t";
      break;
    case CONV_TYPE_LOOP3:
      std::cout << "\tloop_order_3\t";
      break;
    case CONV_TYPE_TILING_LOOP1:
      std::cout << "\ttiling_loop_order_1\t";
      break;
    case CONV_TYPE_TILING_LOOP2:
      std::cout << "\ttiling_loop_order_2\t";
      break;
    case CONV_TYPE_TILING_LOOP3:
      std::cout << "\ttiling_loop_order_3\t";
      break;
    default:
      std::cout << "unknown_type:\t\t";
      break;
  }
  std::cout << std::setprecision(6) << time << " +- "
            << std::setw(10) << error << " ms" << std::endl; // (" << error / time * 100
  //<< "%) -- totally " << total_time << " s in " << n_runs << " runs"
}

void run_time_test_f32(int cycles) {
  const double one_test_time = 0.2;
  double time, error, total_time;
  int n_runs;

  for (int c = 0; c < cycles; ++c) {
    for (int kernel_size: {3, 5}) {
      for (int size: {50, 70, 85, 100}) {
        for (int in_channels: {8, 16}) {
          for (int out_channels: {8, 16}) {
            std::cout << "TESTING convolution of " << size << "x" << size << "x" << in_channels << " image with "
                      << kernel_size << "x" << kernel_size << "x" << in_channels << "x" << out_channels << " kernel"
                      << std::endl;
            ConvWrapper a;
            a.ct = CONV_TYPE_IM2COL_F32;
            a.in_size = MinSize(size, size);
            a.in_channels = in_channels;
            a.filter_size = MinSize(kernel_size, kernel_size);
            a.filters_count = out_channels;
            ConvWrapper b = a;
            ConvWrapper c = a;
            ConvWrapper loop1 = a;
            ConvWrapper loop2 = a;
            ConvWrapper loop3 = a;
            ConvWrapper tiling_loop1 = a;
            ConvWrapper tiling_loop2 = a;
            ConvWrapper tiling_loop3 = a;
            c.ct = CONV_TYPE_ANDERSON_F32;
            b.ct = CONV_TYPE_IM2COL_PARTED_F32;
            loop1.ct = CONV_TYPE_LOOP1;
            loop2.ct = CONV_TYPE_LOOP2;
            loop3.ct = CONV_TYPE_LOOP3;
            tiling_loop1.ct = CONV_TYPE_TILING_LOOP1;
            tiling_loop2.ct = CONV_TYPE_TILING_LOOP2;
            tiling_loop3.ct = CONV_TYPE_TILING_LOOP3;


            loop1.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(loop1, time, error, total_time, n_runs);

            loop2.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(loop2, time, error, total_time, n_runs);

            loop3.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(loop3, time, error, total_time, n_runs);

            tiling_loop1.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(tiling_loop1, time, error, total_time, n_runs);

            tiling_loop2.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(tiling_loop2, time, error, total_time, n_runs);

            tiling_loop3.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(tiling_loop3, time, error, total_time, n_runs);

            a.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(a, time, error, total_time, n_runs);

            c.measure(time, error, total_time, n_runs, one_test_time * 5);
            report(c, time, error, total_time, n_runs);

//            const int out_height = (a.in_size.height - a.filter_size.height + a.stride.height) / a.stride.height;
//            const int out_width = (a.in_size.width - a.filter_size.width + a.stride.width) / a.stride.width;
//            check_correctness(a, c, a.out_channels, out_height, out_width);


            for (int p = 120; p < 121; p += 8) {
              b.parted_rows = p;
              b.measure(time, error, total_time, n_runs, one_test_time);
              report(b, time, error, total_time, n_runs);
            }

          }
        }
      }
    }
  }
}

int main() {
  run_time_test_f32(1);
  return 0;
}
