#include "depth_upscale.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace da {
namespace {

void set_error(std::string* error, const char* message) {
    if (error) *error = message;
}

bool valid_shape(size_t count, int h, int w) {
    return h > 0 && w > 0 && count == static_cast<size_t>(h) * w;
}

int reflect_index(int index, int size) {
    while (index < 0 || index >= size) {
        index = index < 0 ? -index - 1 : 2 * size - index - 1;
    }
    return index;
}

std::vector<float> resize_bilinear_reflect(const std::vector<uint8_t>& image, int in_h, int in_w,
                                           int out_h, int out_w) {
    std::vector<float> out(static_cast<size_t>(out_h) * out_w);
    for (int y = 0; y < out_h; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * in_h / out_h - 0.5f;
        const int y0 = static_cast<int>(std::floor(source_y));
        const float fy = source_y - y0;
        const int y1 = y0 + 1;
        for (int x = 0; x < out_w; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * in_w / out_w - 0.5f;
            const int x0 = static_cast<int>(std::floor(source_x));
            const float fx = source_x - x0;
            const int x1 = x0 + 1;
            const float a = image[static_cast<size_t>(reflect_index(y0, in_h)) * in_w + reflect_index(x0, in_w)];
            const float b = image[static_cast<size_t>(reflect_index(y0, in_h)) * in_w + reflect_index(x1, in_w)];
            const float c = image[static_cast<size_t>(reflect_index(y1, in_h)) * in_w + reflect_index(x0, in_w)];
            const float d = image[static_cast<size_t>(reflect_index(y1, in_h)) * in_w + reflect_index(x1, in_w)];
            out[static_cast<size_t>(y) * out_w + x] =
                (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
        }
    }
    return out;
}

void gaussian_blur_reflect(std::vector<float>& image, int h, int w, float sigma) {
    if (sigma <= 0.0f) return;
    const int radius = static_cast<int>(std::ceil(4.0f * sigma));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float value = std::exp(-0.5f * offset * offset / (sigma * sigma));
        kernel[static_cast<size_t>(offset + radius)] = value;
        sum += value;
    }
    for (float& value : kernel) value /= sum;

    std::vector<float> tmp(image.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float value = 0.0f;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += kernel[static_cast<size_t>(offset + radius)] *
                         image[static_cast<size_t>(y) * w + reflect_index(x + offset, w)];
            }
            tmp[static_cast<size_t>(y) * w + x] = value;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float value = 0.0f;
            for (int offset = -radius; offset <= radius; ++offset) {
                value += kernel[static_cast<size_t>(offset + radius)] *
                         tmp[static_cast<size_t>(reflect_index(y + offset, h)) * w + x];
            }
            image[static_cast<size_t>(y) * w + x] = value;
        }
    }
}

bool solve_linear_system(std::vector<double>& matrix, std::vector<double>& rhs, int n) {
    for (int column = 0; column < n; ++column) {
        int pivot = column;
        for (int row = column + 1; row < n; ++row) {
            if (std::abs(matrix[static_cast<size_t>(row) * n + column]) >
                std::abs(matrix[static_cast<size_t>(pivot) * n + column])) {
                pivot = row;
            }
        }
        const double pivot_value = matrix[static_cast<size_t>(pivot) * n + column];
        if (!std::isfinite(pivot_value) || std::abs(pivot_value) < 1e-12) return false;
        if (pivot != column) {
            for (int j = column; j < n; ++j) {
                std::swap(matrix[static_cast<size_t>(pivot) * n + j], matrix[static_cast<size_t>(column) * n + j]);
            }
            std::swap(rhs[pivot], rhs[column]);
        }
        for (int row = column + 1; row < n; ++row) {
            const double factor = matrix[static_cast<size_t>(row) * n + column] / pivot_value;
            for (int j = column; j < n; ++j) {
                matrix[static_cast<size_t>(row) * n + j] -= factor * matrix[static_cast<size_t>(column) * n + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        double value = rhs[row];
        for (int column = row + 1; column < n; ++column) {
            value -= matrix[static_cast<size_t>(row) * n + column] * rhs[column];
        }
        rhs[row] = value / matrix[static_cast<size_t>(row) * n + row];
        if (!std::isfinite(rhs[row])) return false;
    }
    return true;
}

} // namespace

bool normalize_depth_u8(const std::vector<float>& depth, std::vector<uint8_t>& normalized,
                        std::string* error) {
    if (depth.empty()) {
        set_error(error, "model depth is empty");
        return false;
    }
    auto range = std::minmax_element(depth.begin(), depth.end());
    if (!std::isfinite(*range.first) || !std::isfinite(*range.second) || *range.first >= *range.second) {
        set_error(error, "model depth must contain a finite non-constant range");
        return false;
    }
    const float scale = 255.0f / (*range.second - *range.first);
    normalized.resize(depth.size());
    for (size_t i = 0; i < depth.size(); ++i) {
        normalized[i] = static_cast<uint8_t>(std::clamp((depth[i] - *range.first) * scale, 0.0f, 255.0f));
    }
    return true;
}

bool upscale_depth_map(const std::vector<uint16_t>& sensor_depth, int sensor_h, int sensor_w,
                       const std::vector<uint8_t>& relative_depth, int relative_h, int relative_w,
                       std::vector<uint16_t>& output, const DepthUpscaleOptions& options,
                       std::string* error) {
    if (!valid_shape(sensor_depth.size(), sensor_h, sensor_w) ||
        !valid_shape(relative_depth.size(), relative_h, relative_w)) {
        set_error(error, "depth dimensions do not match pixel counts");
        return false;
    }
    if (options.degree < 0 || options.degree > 8 || !std::isfinite(options.gaussian_sigma) ||
        options.gaussian_sigma < 0.0f) {
        set_error(error, "invalid polynomial degree or Gaussian sigma");
        return false;
    }

    std::vector<float> predictor = resize_bilinear_reflect(relative_depth, relative_h, relative_w, sensor_h, sensor_w);
    gaussian_blur_reflect(predictor, sensor_h, sensor_w, options.gaussian_sigma);
    const float predictor_max = *std::max_element(predictor.begin(), predictor.end());
    const uint16_t sensor_max = *std::max_element(sensor_depth.begin(), sensor_depth.end());
    if (!std::isfinite(predictor_max) || predictor_max <= 0.0f || sensor_max == 0) {
        set_error(error, "sensor and relative depth must have positive maxima");
        return false;
    }

    const int coefficient_count = options.degree + 1;
    std::vector<double> normal(static_cast<size_t>(coefficient_count) * coefficient_count, 0.0);
    std::vector<double> rhs(coefficient_count, 0.0);
    size_t valid_count = 0;
    for (size_t i = 0; i < predictor.size(); ++i) {
        if (sensor_depth[i] == 0 || predictor[i] <= 0.0f || !std::isfinite(predictor[i])) continue;
        const double x = predictor[i] / predictor_max;
        const double y = sensor_depth[i];
        double powers[17] = {1.0};
        for (int p = 1; p <= 2 * options.degree; ++p) powers[p] = powers[p - 1] * x;
        for (int row = 0; row < coefficient_count; ++row) {
            rhs[row] += powers[row] * y;
            for (int column = 0; column < coefficient_count; ++column) {
                normal[static_cast<size_t>(row) * coefficient_count + column] += powers[row + column];
            }
        }
        ++valid_count;
    }
    if (valid_count < static_cast<size_t>(coefficient_count) || !solve_linear_system(normal, rhs, coefficient_count)) {
        set_error(error, "insufficient non-degenerate overlapping depth samples for polynomial fit");
        return false;
    }

    // The original workflow clips to uint16 before its final full-range
    // normalization. Preserve that order so the TIFF spans exactly [0, 65535].
    std::vector<uint16_t> scaled(predictor.size());
    uint16_t output_min = std::numeric_limits<uint16_t>::max();
    uint16_t output_max = 0;
    for (size_t i = 0; i < predictor.size(); ++i) {
        const double x = predictor[i] / predictor_max;
        double value = 0.0;
        for (int coefficient = coefficient_count - 1; coefficient >= 0; --coefficient) {
            value = value * x + rhs[coefficient];
        }
        const uint16_t clipped = static_cast<uint16_t>(std::clamp(value, 0.0, 65535.0));
        scaled[i] = clipped;
        output_min = std::min(output_min, clipped);
        output_max = std::max(output_max, clipped);
    }
    if (output_min >= output_max) {
        set_error(error, "polynomial fit produced a constant or invalid depth map");
        return false;
    }

    const double normalization = 65535.0 / (output_max - output_min);
    output.resize(scaled.size());
    for (size_t i = 0; i < scaled.size(); ++i) {
        output[i] = static_cast<uint16_t>(std::clamp((scaled[i] - output_min) * normalization, 0.0, 65535.0));
    }
    return true;
}

} // namespace da
