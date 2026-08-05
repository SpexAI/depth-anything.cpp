#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace da {

struct DepthUpscaleOptions {
    int degree = 2;
    float gaussian_sigma = 1.0f;
};

// Calibrates a relative Depth Anything map against packed sensor depth. The
// predictor is resized, Gaussian-smoothed, polynomial-fitted over pixels where
// both maps are valid, then normalized to the full uint16 output range.
bool upscale_depth_map(const std::vector<uint16_t>& sensor_depth, int sensor_h, int sensor_w,
                       const std::vector<uint8_t>& relative_depth, int relative_h, int relative_w,
                       std::vector<uint16_t>& output, const DepthUpscaleOptions& options = {},
                       std::string* error = nullptr);

// Matches the former service boundary: render a model depth field to the
// normalized 8-bit image that feeds upscale_depth_map.
bool normalize_depth_u8(const std::vector<float>& depth, std::vector<uint8_t>& normalized,
                        std::string* error = nullptr);

} // namespace da
