#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#if not __cpp_lib_atomic_wait
#include <chrono>
#include <thread>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

// ncnn
#include <net.h>
#include <gpu.h>

#include "onnx2ncnn.hpp"


extern std::variant<std::string, ONNX_NAMESPACE::ModelProto> loadONNX(
    const std::string_view & path,
    int64_t tile_w,
    int64_t tile_h,
    bool path_is_serialization
) noexcept;


static const VSPlugin * myself = nullptr;


[[nodiscard]]
static std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    for (const auto & vi : vis) {
        if (vi->format->sampleType != stFloat || vi->format->bitsPerSample != 32) {
            return "expects clip with type fp32";
        }

        if (vi->width != vis[0]->width || vi->height != vis[0]->height) {
            return "dimensions of clips mismatch";
        }

        if (vi->numFrames != vis[0]->numFrames) {
            return "number of frames mismatch";
        }

        if (vi->format->subSamplingH != 0 || vi->format->subSamplingW != 0) {
            return "clip must not be sub-sampled";
        }
    }

    return {};
}

struct TicketSemaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order_acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order_acquire) };
            if (tk <= curr) {
                return;
            }
#if __cpp_lib_atomic_wait
            current.wait(curr, std::memory_order::relaxed);
#else // __cpp_lib_atomic_wait
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10ms);
#endif // __cpp_lib_atomic_wait
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order_release);
#if __cpp_lib_atomic_wait
        current.notify_all();
#endif // __cpp_lib_atomic_wait
    }
};

// per-stream context
struct Resource {
    std::unique_ptr<ncnn::VkCompute> cmd;
    ncnn::VkAllocator * blob_vkallocator;
    ncnn::VkAllocator * staging_vkallocator;
    ncnn::Mat h_src;
    ncnn::VkMat d_src;
    ncnn::VkMat d_dst;
    ncnn::Mat h_dst;
};

struct vsNcnnData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int overlap_w, overlap_h;

    int in_tile_c, in_tile_w, in_tile_h;
    int out_tile_c, out_tile_w, out_tile_h;

    std::vector<Resource> resources;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    ncnn::VulkanDevice * device; // ncnn caches device allocations in a global variable
    ncnn::Net net;
    int input_index;
    int output_index;

    int acquire() noexcept {
        semaphore.acquire();
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            int ticket = tickets.back();
            tickets.pop_back();
            return ticket;
        }
    }

    void release(int ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            tickets.push_back(ticket);
        }
        semaphore.release();
    }
};


static void VS_CC vsNcnnInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsNcnnData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsNcnnGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsNcnnData *>(*instanceData);

    if (activationReason == arInitial) {
        for (const auto & node : d->nodes) {
            vsapi->requestFrameFilter(n, node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        std::vector<const VSVideoInfo *> in_vis;
        in_vis.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            in_vis.emplace_back(vsapi->getVideoInfo(node));
        }

        std::vector<const VSFrameRef *> src_frames;
        src_frames.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            src_frames.emplace_back(vsapi->getFrameFilter(n, node, frameCtx));
        }

        auto src_stride = vsapi->getStride(src_frames.front(), 0);
        auto src_width = vsapi->getFrameWidth(src_frames.front(), 0);
        auto src_height = vsapi->getFrameHeight(src_frames.front(), 0);
        auto src_bytes = vsapi->getFrameFormat(src_frames.front())->bytesPerSample;

        VSFrameRef * const dst_frame = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames.front(), core
        );
        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;

        auto ticket = d->acquire();
        Resource & resource = d->resources[ticket];

        std::array<int64_t, 4> src_tile_shape { 1, d->in_tile_c, d->in_tile_h, d->in_tile_w };
        auto src_tile_h = src_tile_shape[2];
        auto src_tile_w = src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_tile_shape[1]);
        for (unsigned i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        std::array<int64_t, 4> dst_tile_shape { 1, d->out_tile_c, d->out_tile_h, d->out_tile_w };
        auto dst_tile_h = dst_tile_shape[2];
        auto dst_tile_w = dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_planes = dst_tile_shape[1];
        uint8_t * dst_ptrs[3] {};
        for (int i = 0; i < dst_planes; ++i) {
            dst_ptrs[i] = vsapi->getWritePtr(dst_frame, i);
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        const auto set_error = [&](const std::string & error_message) {
            using namespace std::string_literals;

            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            d->release(ticket);

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        ncnn::Option opt = d->net.opt;
        opt.blob_vkallocator = resource.blob_vkallocator;
        opt.workspace_vkallocator = resource.blob_vkallocator;
        opt.staging_vkallocator = resource.staging_vkallocator;

        int y = 0;
        while (true) {
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    auto input_buffer = reinterpret_cast<uint8_t *>(resource.h_src.data);

                    // assumes the pitches of ncnn::Mat to be
                    // (cstep * elemsize, w * h * elemsize, h * elemsize)
                    for (const auto & _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * src_bytes
                        };

                        {
                            vs_bitblt(
                                input_buffer, src_tile_w_bytes,
                                src_ptr, src_stride,
                                src_tile_w_bytes, src_tile_h
                            );
                            input_buffer += resource.h_src.cstep * sizeof(float);
                        }
                    }
                }

                resource.cmd->record_clone(resource.h_src, resource.d_src, opt);
                if (resource.cmd->submit_and_wait() != 0) {
                    resource.cmd->reset();
                    return set_error("H2D failed");
                }
                if (resource.cmd->reset() != 0) {
                    return set_error("cmd reset failed");
                }

                {
                    auto extractor = d->net.create_extractor();
                    extractor.set_blob_vkallocator(resource.blob_vkallocator);
                    extractor.set_workspace_vkallocator(resource.blob_vkallocator);
                    extractor.set_staging_vkallocator(resource.staging_vkallocator);
                    extractor.input(d->input_index, resource.d_src);
                    extractor.extract(d->output_index, resource.d_dst, *resource.cmd);
                }
                if (resource.cmd->submit_and_wait() != 0) {
                    resource.cmd->reset();
                    return set_error("inference failed");
                }
                if (resource.cmd->reset() != 0) {
                    return set_error("cmd reset failed");
                }

                resource.cmd->record_clone(resource.d_dst, resource.h_dst, opt);
                if (resource.cmd->submit_and_wait() != 0) {
                    resource.cmd->reset();
                    return set_error("D2H failed");
                }
                if (resource.cmd->reset() != 0) {
                    return set_error("cmd reset failed");
                }

                {
                    auto output_buffer = reinterpret_cast<uint8_t *>(resource.h_dst.data);

                    for (int plane = 0; plane < dst_planes; ++plane) {
                        auto dst_ptr = (dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        );

                        {
                            vs_bitblt(
                                dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                                dst_stride,
                                output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                                dst_tile_w_bytes,
                                dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                                dst_tile_h - (y_crop_start + y_crop_end)
                            );

                            output_buffer += resource.h_dst.cstep * sizeof(float);
                        }
                    }
                }

                if (x + src_tile_w == src_width) {
                    break;
                }

                x = std::min(x + step_w, src_width - src_tile_w);
            }

            if (y + src_tile_h == src_height) {
                break;
            }

            y = std::min(y + step_h, src_height - src_tile_h);
        }

        d->release(ticket);

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsNcnnFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsNcnnData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    for (const auto & resource : d->resources) {
        d->device->reclaim_blob_allocator(resource.blob_vkallocator);
        d->device->reclaim_staging_allocator(resource.staging_vkallocator);
    }

    delete d;
}


static void VS_CC vsNcnnCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsNcnnData>() };

    int num_nodes = vsapi->propNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->propGetNode(in, "clips", i, nullptr));
    }

    auto set_error = [&](const std::string & error_message) {
        using namespace std::string_literals;
        vsapi->setError(out, (__func__ + ": "s + error_message).c_str());
        for (const auto & node : d->nodes) {
            vsapi->freeNode(node);
        }
    };

    std::vector<const VSVideoInfo *> in_vis;
    in_vis.reserve(std::size(d->nodes));
    for (const auto & node : d->nodes) {
        in_vis.emplace_back(vsapi->getVideoInfo(node));
    }

    if (auto err = checkNodes(in_vis); err.has_value()) {
        return set_error(err.value());
    }

    int error;

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    int error1, error2;
    d->overlap_w = int64ToIntS(vsapi->propGetInt(in, "overlap", 0, &error1));
    d->overlap_h = int64ToIntS(vsapi->propGetInt(in, "overlap", 1, &error2));
    if (!error1) {
        if (error2) {
            d->overlap_h = d->overlap_w;
        }

        if (d->overlap_w < 0 || d->overlap_h < 0) {
            return set_error("\"overlap\" must be non-negative");
        }
    } else {
        d->overlap_w = 0;
        d->overlap_h = 0;
    }

    int64_t tile_w = vsapi->propGetInt(in, "tilesize", 0, &error1);
    int64_t tile_h = vsapi->propGetInt(in, "tilesize", 1, &error2);
    if (!error1) { // manual specification triggered
        if (error2) {
            tile_h = tile_w;
        }
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("\"tilesize\" must be specified");
        }

        // set tile size to video dimensions
        tile_w = in_vis.front()->width;
        tile_h = in_vis.front()->height;
    }
    if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
        return set_error("\"overlap\" too large");
    }

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

    d->semaphore.current.store(num_streams - 1, std::memory_order_relaxed);
    d->tickets.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        d->tickets.push_back(i);
    }

    bool fp16 = !!vsapi->propGetInt(in, "fp16", 0, &error);
    if (error) {
        fp16 = false;
    }

    bool path_is_serialization = !!vsapi->propGetInt(in, "path_is_serialization", 0, &error);
    if (error) {
        path_is_serialization = false;
    }

    std::string_view path_view;
    std::string path;
    if (path_is_serialization) {
        path_view = {
            vsapi->propGetData(in, "network_path", 0, nullptr),
            static_cast<size_t>(vsapi->propGetDataSize(in, "network_path", 0, nullptr))
        };
    } else {
        path = vsapi->propGetData(in, "network_path", 0, nullptr);
        bool builtin = !!vsapi->propGetInt(in, "builtin", 0, &error);
        if (builtin) {
            const char *modeldir = vsapi->propGetData(in, "builtindir", 0, &error);
            if (!modeldir) modeldir = "models";
            path = std::string(modeldir) + "/" + path;
            std::string dir { vsapi->getPluginPath(myself) };
            dir = dir.substr(0, dir.rfind('/') + 1);
            path = dir + path;
        }
        path_view = path;
    }

    auto result = loadONNX(path_view, tile_w, tile_h, path_is_serialization);
    if (std::holds_alternative<std::string>(result)) {
        return set_error(std::get<std::string>(result));
    }

    auto onnx_model = std::move(std::get<ONNX_NAMESPACE::ModelProto>(result));
    {
        const auto & input_shape = onnx_model.graph().input(0).type().tensor_type().shape();
        d->in_tile_c = int64ToIntS(input_shape.dim(1).dim_value());
        d->in_tile_h = int64ToIntS(input_shape.dim(2).dim_value());
        d->in_tile_w = int64ToIntS(input_shape.dim(3).dim_value());

        const auto & output_shape = onnx_model.graph().output(0).type().tensor_type().shape();
        d->out_tile_c = int64ToIntS(output_shape.dim(1).dim_value());
        d->out_tile_h = int64ToIntS(output_shape.dim(2).dim_value());
        d->out_tile_w = int64ToIntS(output_shape.dim(3).dim_value());
    }

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis.front()); // mutable
    d->out_vi->width *= d->out_tile_w / d->in_tile_w;
    d->out_vi->height *= d->out_tile_h / d->in_tile_h;
    if (d->out_tile_c == 1) {
        d->out_vi->format = vsapi->registerFormat(cmGray, stFloat, 32, 0, 0, core);
    }
    if (d->out_tile_c == 3) {
        d->out_vi->format = vsapi->registerFormat(cmRGB, stFloat, 32, 0, 0, core);
    }

    auto ncnn_result = onnx2ncnn(onnx_model);
    if (!ncnn_result.has_value()) {
        return set_error("onnx2ncnn failed");
    }

    const auto & [ncnn_param, ncnn_model_bin] = ncnn_result.value();

    // ncnn related code
    if (auto device = ncnn::get_gpu_device(device_id); device != nullptr) {
        d->device = device;
    } else {
        vs_aligned_free(ncnn_param);
        vs_aligned_free(ncnn_model_bin);
        return set_error("get_gpu_device failed");
    }

    d->net.opt.num_threads = 1;
    d->net.opt.use_vulkan_compute = true;
    d->net.opt.use_fp16_packed = fp16;
    d->net.opt.use_fp16_storage = fp16;
    d->net.opt.use_fp16_arithmetic = fp16;
    d->net.opt.use_int8_packed = false;
    d->net.opt.use_int8_storage = false;
    d->net.opt.use_int8_arithmetic = false;
    d->net.set_vulkan_device(d->device);
    if (d->net.load_param_mem(ncnn_param) != 0) {
        vs_aligned_free(ncnn_param);
        vs_aligned_free(ncnn_model_bin);
        return set_error("load param failed");
    }
    vs_aligned_free(ncnn_param);
    // TODO: here returns the number of bytes read successfully
    d->net.load_model(ncnn_model_bin);
    vs_aligned_free(ncnn_model_bin);

    d->input_index = d->net.input_indexes().front();
    d->output_index = d->net.output_indexes().front();

    d->resources.resize(num_streams);
    for (auto & resource : d->resources) {
        resource.cmd = std::make_unique<ncnn::VkCompute>(d->device);
        resource.blob_vkallocator = d->device->acquire_blob_allocator();
        resource.staging_vkallocator = d->device->acquire_staging_allocator();
        resource.h_src.create(d->in_tile_w, d->in_tile_h, d->in_tile_c);
        resource.d_src.create(d->in_tile_w, d->in_tile_h, d->in_tile_c, sizeof(float), resource.blob_vkallocator);
        resource.d_dst.create(d->out_tile_w, d->out_tile_h, d->out_tile_c, sizeof(float), resource.blob_vkallocator);
        resource.h_dst.create(d->out_tile_w, d->out_tile_h, d->out_tile_c);
    }

    vsapi->createFilter(
        in, out, "Model",
        vsNcnnInit, vsNcnnGetFrame, vsNcnnFree,
        fmParallel, 0, d.release(), core
    );
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin *plugin
) noexcept {

    myself = plugin;

    configFunc(
        "io.github.amusementclub.vs_ncnn", "ncnn",
        "NCNN ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "network_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "builtin:int:opt;"
        "builtindir:data:opt;"
        "fp16:int:opt;"
        "path_is_serialization:int:opt;"
        , vsNcnnCreate,
        nullptr,
        plugin
    );
}