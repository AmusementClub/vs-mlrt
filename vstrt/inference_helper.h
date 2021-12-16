#ifndef VSTRT_INFERENCE_HELPER_H_
#define VSTRT_INFERENCE_HELPER_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <VSHelper.h>

#include "cuda_helper.h"
#include "trt_utils.h"

struct InputInfo {
    int width;
    int height;
    int pitch;
    int bytes_per_sample;
    int tile_w;
    int tile_h;
};

struct OutputInfo {
    int pitch;
    int bytes_per_sample;
};

struct IOInfo {
    InputInfo in;
    OutputInfo out;
    int w_scale;
    int h_scale;
    int overlap_w;
    int overlap_h;
};

static inline
std::optional<ErrorMessage> inference(
    const InferenceInstance & instance,
    int device_id,
    bool use_cuda_graph, 
    const IOInfo & info,
    const std::vector<const uint8_t *> & src_ptrs,
    const std::vector<uint8_t *> & dst_ptrs
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    checkError(cudaSetDevice(device_id));

    int src_tile_w_bytes = info.in.tile_w * info.in.bytes_per_sample;
    int src_tile_bytes = info.in.tile_h * info.in.tile_w * info.in.bytes_per_sample;
    int dst_tile_w = info.in.tile_w * info.w_scale;
    int dst_tile_h = info.in.tile_h * info.h_scale;
    int dst_tile_w_bytes = dst_tile_w * info.out.bytes_per_sample;
    int dst_tile_bytes = dst_tile_h * dst_tile_w * info.out.bytes_per_sample;

    int step_w = info.in.tile_w - 2 * info.overlap_w;
    int step_h = info.in.tile_h - 2 * info.overlap_h;

    int y = 0;
    while (true) {
        int y_crop_start = (y == 0) ? 0 : info.overlap_h;
        int y_crop_end = (y == info.in.height - info.in.tile_h) ? 0 : info.overlap_h;

        int x = 0;
        while (true) {
            int x_crop_start = (x == 0) ? 0 : info.overlap_w;
            int x_crop_end = (x == info.in.width - info.in.tile_w) ? 0 : info.overlap_w;

            {
                uint8_t * h_data = instance.src.h_data.data;
                for (const uint8_t * _src_ptr : src_ptrs) {
                    const uint8_t * src_ptr { _src_ptr +
                        y * info.in.pitch + x * info.in.bytes_per_sample
                    };

                    vs_bitblt(
                        h_data, src_tile_w_bytes,
                        src_ptr, info.in.pitch,
                        src_tile_w_bytes, info.in.tile_h
                    );

                    h_data += src_tile_bytes;
                }
            }

            if (use_cuda_graph) {
                checkError(cudaGraphLaunch(instance.graphexec, instance.stream));
            } else {
                auto result = enqueue(
                    instance.src, instance.dst,
                    instance.exec_context, instance.stream
                );

                if (result.has_value()) {
                    return set_error(result.value());
                }
            }
            checkError(cudaStreamSynchronize(instance.stream));

            {
                const uint8_t * h_data = instance.dst.h_data.data;
                for (uint8_t * _dst_ptr : dst_ptrs) {
                    uint8_t * dst_ptr { _dst_ptr +
                        info.h_scale * y * info.out.pitch + info.w_scale * x * info.out.bytes_per_sample
                    };

                    vs_bitblt(
                        dst_ptr + (y_crop_start * info.out.pitch + x_crop_start * info.out.bytes_per_sample),
                        info.out.pitch,
                        h_data + (y_crop_start * dst_tile_w_bytes + x_crop_start * info.out.bytes_per_sample),
                        dst_tile_w_bytes,
                        dst_tile_w_bytes - (x_crop_start + x_crop_end) * info.out.bytes_per_sample,
                        dst_tile_h - (y_crop_start + y_crop_end)
                    );

                    h_data += dst_tile_bytes;
                }
            }

            if (x + info.in.tile_w == info.in.width) {
                break;
            }

            x = std::min(x + step_w, info.in.width - info.in.tile_w);
        }

        if (y + info.in.tile_h == info.in.height) {
            break;
        }

        y = std::min(y + step_h, info.in.height - info.in.tile_h);
    }

    return {};
}

#endif // VSTRT_INFERENCE_HELPER_H_
