#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <iree/runtime/api.h>

#include "config.h"

using namespace std::string_literals;

#define checkError(expr) do {                                \
    using namespace std::string_literals;                    \
    iree_status_t status = expr;                             \
    if (!iree_status_is_ok(status)) {                        \
        auto message = iree::Status().ToString(status);      \
        return set_error("'" # expr "' failed: " + message); \
    }                                                        \
} while(0)

static const VSPlugin * myself = nullptr;

// get an iree instance; it will not be explicitly released
static
iree_runtime_instance_t * get_iree_instance() {
    static std::atomic<iree_runtime_instance_t *> global_instance {};

    auto ret = global_instance.load(std::memory_order_relaxed);
    if (ret == nullptr) {
        static std::mutex instance_lock;
        std::scoped_lock _(instance_lock);
        ret = global_instance.load(std::memory_order_relaxed);
        if (ret == nullptr) {
            iree_runtime_instance_options_t instance_options;
            iree_runtime_instance_options_initialize(&instance_options);
            iree_runtime_instance_options_use_all_available_drivers(&instance_options);
            iree_status_t status = iree_runtime_instance_create(
                &instance_options,
                iree_allocator_system(),
                &ret
            );
            if (iree_status_is_ok(status)) {
                global_instance.store(ret, std::memory_order_relaxed);
            } else {
                fprintf(stderr, "vsiree: ");
                iree_status_fprint(stderr, status);
            }
        }
    }
    return ret;
}

static
void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::array<int, 4> & input_shape,
    const std::array<int, 4> & output_shape,
    int bitsPerSample,
    VSCore * core,
    const VSAPI * vsapi,
    bool flexible_output
) noexcept {

    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];

    if (output_shape[1] == 1 || flexible_output) {
        vi->format = vsapi->registerFormat(cmGray, stFloat, bitsPerSample, 0, 0, core);
    } else if (output_shape[1] == 3) {
        vi->format = vsapi->registerFormat(cmRGB, stFloat, bitsPerSample, 0, 0, core);
    }
}

static inline
std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    for (const auto & vi : vis) {
        if (!isConstantFormat(vi)) {
            return "video format must be constant";
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

        if (vi->format->sampleType != vis[0]->format->sampleType) {
            return "sample type mismatch";
        }

        if (vi->format->bitsPerSample != vis[0]->format->bitsPerSample) {
            return "bits per sample mismatch";
        }
    }

    return {};
}

static inline
int numPlanes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    int num_planes = 0;

    for (const auto & vi : vis) {
        num_planes += vi->format->numPlanes;
    }

    return num_planes;
}

struct TicketSemaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void init(intptr_t num) noexcept {
        current.store(num, std::memory_order::seq_cst);
    }

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk < curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
    requires
        std::default_initializable<T> &&
        std::movable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]]
    constexpr Resource() noexcept = default;

    [[nodiscard]]
    constexpr Resource(T && x) noexcept : data(x) {}

    [[nodiscard]]
    constexpr Resource(Resource&& other) noexcept
        : data(std::exchange(other.data, T{}))
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(std::move(data));
        data = std::exchange(other.data, T{});
        return *this;
    }

    constexpr Resource& operator=(const Resource & other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T && x) noexcept {
        if (x) {
            (void) deleter(x);
        }
    }

    constexpr Resource& operator=(T && x) noexcept {
        deleter_(std::move(data));
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(std::move(data));
    }
};

static
std::expected<std::array<int, 4>, std::string> get_output_shape(iree_runtime_session_t * session, std::array<int, 4> input_shape) {
    const auto set_error = [](const std::string & error_message) {
        return std::unexpected(error_message);
    };

    iree_runtime_call_t call;
    checkError(iree_runtime_call_initialize_by_name(
        session,
        iree_make_cstring_view("module.tf2onnx"),
        &call
    ));
    auto device = iree_runtime_session_device(session);
    auto device_allocator = iree_runtime_session_device_allocator(session);
    const iree_hal_dim_t shape[4] = {
        static_cast<iree_hal_dim_t>(input_shape[0]),
        static_cast<iree_hal_dim_t>(input_shape[1]),
        static_cast<iree_hal_dim_t>(input_shape[2]),
        static_cast<iree_hal_dim_t>(input_shape[3])
    };
    Resource<iree_hal_buffer_view_t *, iree_hal_buffer_view_release> input {};
    checkError(iree_hal_buffer_view_allocate_buffer_copy(
        device,
        device_allocator,
        IREE_ARRAYSIZE(input_shape),
        shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t) {
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_const_byte_span_empty(),
        &input.data
    ));
    checkError(iree_runtime_call_inputs_push_back_buffer_view(&call, input));

    checkError(iree_runtime_call_invoke(&call, 0));

    Resource<iree_hal_buffer_view_t *, iree_hal_buffer_view_release> output {};
    checkError(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output.data));

    if (iree_hal_buffer_view_shape_rank(output) != 4) {
        return set_error("invalid shape dim");
    }

    auto dims = iree_hal_buffer_view_shape_dims(output);
    return std::array{
        static_cast<int>(dims[0]),
        static_cast<int>(dims[1]),
        static_cast<int>(dims[2]),
        static_cast<int>(dims[3]),
    };
}

struct MemoryResource {
    // Resource<uint8_t *, hipHostFree> h_data;
    // Resource<uint8_t *, hipFree> d_data;
    Resource<iree_hal_buffer_t *, iree_hal_buffer_destroy> d_buffer;
    // Resource<iree_hal_buffer_view_t *, iree_hal_buffer_view_destroy> d_buffer_view;
    size_t size;
};

struct InferenceInstance {
    Resource<iree_runtime_session_t *, iree_runtime_session_release> session;
    MemoryResource src;
    // MemoryResource dst;
    // Resource<migraphx_program_parameters_t, migraphx_program_parameters_destroy> params;
    // Resource<migraphx_argument_t, migraphx_argument_destroy> src_argument;
    // Resource<migraphx_argument_t, migraphx_argument_destroy> dst_argument;
    // Resource<hipStream_t, hipStreamDestroy> stream;
};


static
std::expected<InferenceInstance, std::string> create_instance(
    iree_runtime_instance_t * iree_instance,
    const char * module_path,
    std::array<iree_hal_dim_t, 4> input_shape
) {

    const auto set_error = [](const std::string & error_message) {
        return std::unexpected(error_message);
    };

    InferenceInstance instance;

    Resource<iree_hal_device_t *, iree_hal_device_release> device {};

    checkError(iree_runtime_instance_try_create_default_device(
        iree_instance,
        iree_make_cstring_view("local-task"),
        &device.data
    ));

    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    checkError(iree_runtime_session_create_with_device(
        iree_instance,
        &session_options,
        device,
        iree_runtime_instance_host_allocator(iree_instance),
        &instance.session.data
    ));
    checkError(iree_runtime_session_append_bytecode_module_from_file(instance.session, module_path));

    iree_hal_buffer_params_t buffer_params {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };
    iree_hal_buffer_params_canonicalize(&buffer_params);

    iree_device_size_t allocation_size;
    checkError(iree_hal_buffer_compute_view_size(
        IREE_ARRAYSIZE(input_shape),
        input_shape.data(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        &allocation_size
    ));

    auto device_allocator = iree_runtime_session_device_allocator(instance.session);
    checkError(iree_hal_allocator_allocate_buffer(
        device_allocator,
        buffer_params,
        allocation_size,
        &instance.src.d_buffer.data
    ));

    return instance;
}

struct vsIREEData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    std::array<int, 4> src_tile_shape, dst_tile_shape;
    int overlap_w, overlap_h;

    std::vector<InferenceInstance> instances;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    std::string flexible_output_prop;

    [[nodiscard]] int acquire() noexcept {
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


static void VS_CC vsIREEInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsIREEData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsIREEGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsIREEData *>(*instanceData);

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

        std::vector<VSFrameRef *> dst_frames;

        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;

        auto ticket = d->acquire();
        InferenceInstance & instance = d->instances[ticket];

        auto src_tile_h = d->src_tile_shape[2];
        auto src_tile_w = d->src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;
        auto src_tile_bytes = src_tile_h * src_tile_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(d->src_tile_shape[1]);
        for (unsigned i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        auto dst_tile_h = d->dst_tile_shape[2];
        auto dst_tile_w = d->dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_tile_bytes = dst_tile_h * dst_tile_w_bytes;
        auto dst_planes = d->dst_tile_shape[1];

        std::vector<uint8_t *> dst_ptrs;
        if (d->flexible_output_prop.empty()) {
            for (int i = 0; i < dst_planes; ++i) {
                dst_ptrs.emplace_back(vsapi->getWritePtr(dst_frame, i));
            }
        } else {
            for (int i = 0; i < dst_planes; ++i) {
                auto frame { vsapi->newVideoFrame(
                    d->out_vi->format, d->out_vi->width, d->out_vi->height,
                    src_frames[0], core
                )};
                dst_frames.emplace_back(frame);
                dst_ptrs.emplace_back(vsapi->getWritePtr(frame, 0));
            }
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        const auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            d->release(ticket);

            for (const auto & frame : dst_frames) {
                vsapi->freeFrame(frame);
            }

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        auto & session = instance.session;
        iree_runtime_call_t call;
        checkError(iree_runtime_call_initialize_by_name(
            session,
            iree_make_cstring_view("module.tf2onnx"),
            &call
        ));
        auto device = iree_runtime_session_device(session);
        auto host_allocator = iree_runtime_session_host_allocator(session);

        int y = 0;
        while (true) {
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    // uint8_t * h_data = instance.src.h_data.data;
                    // for (const uint8_t * _src_ptr : src_ptrs) {
                    //     const uint8_t * src_ptr { _src_ptr +
                    //         y * src_stride + x * vsapi->getFrameFormat(src_frames[0])->bytesPerSample
                    //     };

                    //     vs_bitblt(
                    //         h_data, src_tile_w_bytes,
                    //         src_ptr, src_stride,
                    //         src_tile_w_bytes, src_tile_h
                    //     );

                    //     h_data += src_tile_bytes;
                    // }
                }

                // checkHIPError(hipMemcpyAsync(
                //     instance.src.d_data.data,
                //     instance.src.h_data.data,
                //     instance.src.size,
                //     hipMemcpyHostToDevice,
                //     instance.stream
                // ));

                {
                    const iree_hal_dim_t input_shape[4] = {
                        static_cast<iree_hal_dim_t>(d->src_tile_shape[0]),
                        static_cast<iree_hal_dim_t>(d->src_tile_shape[1]),
                        static_cast<iree_hal_dim_t>(d->src_tile_shape[2]),
                        static_cast<iree_hal_dim_t>(d->src_tile_shape[3])
                    };
                    checkError(iree_hal_device_transfer_h2d(
                        device,
                        src_ptrs[0],
                        instance.src.d_buffer,
                        0,
                        d->src_tile_shape[0] * d->src_tile_shape[1] * 
                        d->src_tile_shape[2] * d->src_tile_shape[3] * sizeof(float),
                        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                        iree_infinite_timeout()
                    ));
                    Resource<iree_hal_buffer_view_t *, iree_hal_buffer_view_release> input;
                    checkError(iree_hal_buffer_view_create(
                        instance.src.d_buffer,
                        IREE_ARRAYSIZE(input_shape),
                        input_shape,
                        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                        host_allocator,
                        &input.data
                    ));
                    checkError(iree_runtime_call_inputs_push_back_buffer_view(&call, input));
                }

                checkError(iree_runtime_call_invoke(&call, 0));

                Resource<iree_hal_buffer_view_t *, iree_hal_buffer_view_release> output {};
                checkError(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output.data));
                auto buffer = iree_hal_buffer_view_buffer(output);
                iree_hal_buffer_map_read(
                    buffer,
                    0,
                    dst_ptrs[0],
                    d->dst_tile_shape[0] * d->dst_tile_shape[1] *
                    d->dst_tile_shape[2] * d->dst_tile_shape[3] * sizeof(float)
                );

                {
                    // const uint8_t * h_data = instance.dst.h_data.data;
                    // auto bytes_per_sample = vsapi->getFrameFormat(dst_frame)->bytesPerSample;
                    // for (int plane = 0; plane < dst_planes; ++plane) {
                    //     uint8_t * dst_ptr {
                    //         dst_ptrs[plane] +
                    //         h_scale * y * dst_stride + w_scale * x * dst_bytes
                    //     };

                    //     vs_bitblt(
                    //         dst_ptr + (y_crop_start * dst_stride + x_crop_start * bytes_per_sample),
                    //         dst_stride,
                    //         h_data + (y_crop_start * dst_tile_w_bytes + x_crop_start * bytes_per_sample),
                    //         dst_tile_w_bytes,
                    //         dst_tile_w_bytes - (x_crop_start + x_crop_end) * bytes_per_sample,
                    //         dst_tile_h - (y_crop_start + y_crop_end)
                    //     );

                    //     h_data += dst_tile_bytes;
                    // }
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

        iree_runtime_call_deinitialize(&call);

        d->release(ticket);

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        if (!d->flexible_output_prop.empty()) {
            auto prop = vsapi->getFramePropsRW(dst_frame);

            for (int i = 0; i < dst_planes; i++) {
                auto key { d->flexible_output_prop + std::to_string(i) };
                vsapi->propSetFrame(prop, key.c_str(), dst_frames[i], paReplace);
                vsapi->freeFrame(dst_frames[i]);
            }
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsIREEFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsIREEData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    delete d;
}


static void VS_CC vsIREECreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsIREEData>() };

    int num_nodes = vsapi->propNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->propGetNode(in, "clips", i, nullptr));
    }

    auto set_error = [&](const std::string & error_message) {
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

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis.front()); // mutable


    int error;

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

    size_t tile_w = static_cast<size_t>(vsapi->propGetInt(in, "tilesize", 0, &error1));
    size_t tile_h = static_cast<size_t>(vsapi->propGetInt(in, "tilesize", 1, &error2));
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

    auto flexible_output_prop = vsapi->propGetData(in, "flexible_output_prop", 0, &error);
    if (!error) {
        d->flexible_output_prop = flexible_output_prop;
    }

    auto iree_instance = get_iree_instance();
    if (!iree_instance) {
        return set_error("get iree instance failed");
    }

    // {
    //     migraphx_file_options_t file_options;
    //     checkError(migraphx_file_options_create(&file_options));
    //     const char * program_path = vsapi->propGetData(in, "program_path", 0, nullptr);
    //     checkError(migraphx_load(&d->program, program_path, file_options));
    //     checkError(migraphx_file_options_destroy(file_options));
    // }
    // checkHIPError(hipSetDevice(d->device_id));

//     const char * input_name[2];
//     const_migraphx_shape_t input_shape;
//     size_t input_size;
//     {
//         migraphx_program_parameter_shapes_t input_shapes;
//         checkError(migraphx_program_get_parameter_shapes(&input_shapes, d->program));
//         size_t num_inputs;
//         checkError(migraphx_program_parameter_shapes_size(&num_inputs, input_shapes));
//         if (num_inputs != 2) {
//             return set_error("program must have exactly one input");
//         }
//         checkError(migraphx_program_parameter_shapes_names(&input_name[0], input_shapes));
//         // here we assume that the second parameter corresponds to the input node
//         checkError(migraphx_program_parameter_shapes_get(&input_shape, input_shapes, input_name[1]));
//         // TODO: support dynamic shapes
// #ifdef MIGRAPHX_VERSION_TWEAK
//         bool is_dynamic;
//         checkError(migraphx_shape_dynamic(&is_dynamic, input_shape));
//         // TODO
//         if (is_dynamic) {
//             return set_error("dynamic shape is not supported for now");
//         }
// #endif // MIGRAPHX_VERSION_TWEAK
//         migraphx_shape_datatype_t type;
//         checkError(migraphx_shape_type(&type, input_shape));
//         if (type != migraphx_shape_float_type && type != migraphx_shape_half_type) {
//             return set_error("input type must be float or half");
//         }
//         if (in_vis[0]->format->sampleType != getSampleType(type)) {
//             return set_error("sample type mismatch");
//         }
//         if (in_vis[0]->format->bytesPerSample != getBytesPerSample(type)) {
//             return set_error("bytes per sample mismatch");
//         }
//         const size_t * lengths;
//         size_t ndim;
//         checkError(migraphx_shape_lengths(&lengths, &ndim, input_shape));
//         if (ndim != 4) {
//             return set_error("number of input dimension must be 4");
//         }
//         if (lengths[0] != 1) {
//             return set_error("batch size must be 1");
//         }
//         if (auto num_planes = numPlanes(in_vis); static_cast<int>(lengths[1]) != num_planes) {
//             return set_error("expects " + std::to_string(lengths[1]) + " input planes");
//         }
//         // TODO: select
//         if (lengths[2] != tile_h || lengths[3] != tile_w) {
//             return set_error(
//                 "invalid tile size, must be " +
//                 std::to_string(lengths[3]) + 'x' + std::to_string(lengths[2])
//             );
//         }
//         const size_t * strides;
//         checkError(migraphx_shape_strides(&strides, &ndim, input_shape));
//         {
//             size_t target = 1; // IREE uses elements to measure strides
//             for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
//                 if (strides[i] != target) {
//                     return set_error(
//                         "invalid stride for NCHW, expects " +
//                         std::to_string(target) +
//                         " instead of " +
//                         std::to_string(strides[i])
//                     );
//                 }
//                 target *= lengths[i];
//             }
//         }
//         checkError(migraphx_shape_bytes(&input_size, input_shape));
//         for (int i = 0; i < 4; i++) {
//             d->src_tile_shape[i] = static_cast<int>(lengths[i]);
//         }
//     }

//     size_t output_size;
//     const_migraphx_shape_t output_shape;
//     int bitsPerSample;
//     {
//         migraphx_shapes_t output_shapes;
//         checkError(migraphx_program_get_output_shapes(&output_shapes, d->program));
//         size_t num_outputs;
//         checkError(migraphx_shapes_size(&num_outputs, output_shapes));
//         if (num_outputs != 1) {
//             return set_error("program must have exactly one output");
//         }
//         checkError(migraphx_shapes_get(&output_shape, output_shapes, 0));
//         // TODO: support dynamic shapes
// #ifdef MIGRAPHX_VERSION_TWEAK
//         bool is_dynamic;
//         checkError(migraphx_shape_dynamic(&is_dynamic, output_shape));
//         // TODO
//         if (is_dynamic) {
//             return set_error("dynamic shape is not supported for now");
//         }
// #endif // MIGRAPHX_VERSION_TWEAK
//         migraphx_shape_datatype_t type;
//         checkError(migraphx_shape_type(&type, output_shape));
//         if (type != migraphx_shape_float_type && type != migraphx_shape_half_type) {
//             return set_error("output type must be float or half");
//         }
//         bitsPerSample = type == migraphx_shape_float_type ? 32 : 16;
//         const size_t * lengths;
//         size_t ndim;
//         checkError(migraphx_shape_lengths(&lengths, &ndim, output_shape));
//         if (ndim != 4) {
//             return set_error("number of output dimension must be 4");
//         }
//         if (lengths[0] != 1) {
//             return set_error("batch size must be 1");
//         }
//         if (lengths[1] != 1 && lengths[1] != 3 && d->flexible_output_prop.empty()) {
//             return set_error("output should have 1 or 3 channels, or enable \"flexible_output\"");
//         }
//         if (lengths[2] % tile_h != 0 && lengths[3] % tile_w != 0) {
//             return set_error("output dimensions should be integer multiple of input dimensions");
//         }
//         const size_t * strides;
//         checkError(migraphx_shape_strides(&strides, &ndim, output_shape));
//         {
//             size_t target = 1; // IREE uses elements to measure strides
//             for (int i = static_cast<int>(ndim) - 1; i >= 0; i--) {
//                 if (strides[i] != target) {
//                     return set_error(
//                         "invalid stride for NCHW, expects " +
//                         std::to_string(target) +
//                         " instead of " +
//                         std::to_string(strides[i])
//                     );
//                 }
//                 target *= lengths[i];
//             }
//         }
//         checkError(migraphx_shape_bytes(&output_size, output_shape));
//         for (int i = 0; i < 4; i++) {
//             d->dst_tile_shape[i] = static_cast<int>(lengths[i]);
//         }
//     }
    d->src_tile_shape[0] = 1;
    d->src_tile_shape[1] = numPlanes(in_vis);
    d->src_tile_shape[2] = tile_h;
    d->src_tile_shape[3] = tile_w;

    int num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }
    if (num_streams <= 0) {
        return set_error("\"num_streams\" must be positive");
    }

    // per-stream context
    d->instances.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        auto input_shape = std::array{
            static_cast<iree_hal_dim_t>(d->src_tile_shape[0]),
            static_cast<iree_hal_dim_t>(d->src_tile_shape[1]),
            static_cast<iree_hal_dim_t>(d->src_tile_shape[2]),
            static_cast<iree_hal_dim_t>(d->src_tile_shape[3])
        };
        auto result = create_instance(
            iree_instance,
            vsapi->propGetData(in, "module_path", 0, nullptr),
            input_shape
        );
        if (result.has_value()) {
            d->instances.emplace_back(std::move(result.value()));
        } else {
            return set_error(result.error());
        }
    }

    auto shape = get_output_shape(d->instances[0].session, d->src_tile_shape);
    if (shape.has_value()) {
        d->dst_tile_shape = shape.value();
    } else {
        return set_error(shape.error());
    }

    setDimensions(
        d->out_vi,
        d->src_tile_shape,
        d->dst_tile_shape,
        32,
        core,
        vsapi,
        !d->flexible_output_prop.empty()
    );

    d->semaphore.init(num_streams);
    d->tickets.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        d->tickets.push_back(i);
    }

    if (!d->flexible_output_prop.empty()) {
        vsapi->propSetInt(out, "num_planes", d->dst_tile_shape[1], paReplace);
    }

    vsapi->createFilter(
        in, out, "Model",
        vsIREEInit, vsIREEGetFrame, vsIREEFree,
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
        "io.github.amusementclub.vs_iree", "iree",
        "IREE ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "module_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "flexible_output_prop:data:opt;"
        , vsIREECreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);

    // registerFunc("DeviceProperties", "device_id:int:opt;", getDeviceProp, nullptr, plugin);
}
