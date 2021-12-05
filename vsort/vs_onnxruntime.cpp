#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <onnx/proto_utils.h>
#include <onnxruntime_c_api.h>

#include <VapourSynth.h>
#include <VSHelper.h>

#ifdef _WIN32
#include <locale>
#include <codecvt>
static inline std::wstring translateName(const char *name) noexcept {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(name);
}
#else
#define translateName(n) (n)
#endif

#define checkError(expr) do {                                                  \
    OrtStatusPtr __err = expr;                                                 \
    if (__err) {                                                               \
        const std::string message = ortapi->GetErrorMessage(__err);            \
        ortapi->ReleaseStatus(__err);                                          \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

using namespace std::string_literals;


static const OrtApi * ortapi = nullptr;

[[nodiscard]]
static std::optional<std::string> ortInit() noexcept {
    static std::once_flag ort_init_flag;

    std::call_once(ort_init_flag, []() {
        ortapi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    });

    if (ortapi) {
        return {};
    } else {
        return "ONNX Runtime initialization failed";
    }
}


[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(
    const OrtTensorTypeAndShapeInfo* tensor_info
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    std::array<int64_t, 4> shape;
    checkError(ortapi->GetDimensions(tensor_info, shape.data(), std::size(shape)));

    return shape;
}


[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(
    const OrtSession * session,
    bool input
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    OrtTypeInfo * typeinfo;
    if (input) {
        checkError(ortapi->SessionGetInputTypeInfo(session, 0, &typeinfo));
    } else {
        checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &typeinfo));
    }

    const OrtTensorTypeAndShapeInfo* tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    auto maybe_shape = getShape(tensor_info);
    ortapi->ReleaseTypeInfo(typeinfo);

    if (std::holds_alternative<std::string>(maybe_shape)) {
        return set_error(std::get<std::string>(maybe_shape));
    }

    return std::get<std::array<int64_t, 4>>(maybe_shape);
}


static void specifyShape(
    onnx::ModelProto & model,
    int64_t block_w,
    int64_t block_h,
    int64_t batch = 1
) noexcept {

    onnx::TensorShapeProto * input_shape {
        model
            .mutable_graph()
            ->mutable_input()
            ->begin()
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };
    onnx::TensorShapeProto * output_shape {
        model
            .mutable_graph()
            ->mutable_output()
            ->begin()
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };

    constexpr auto n_idx = 0;
    constexpr auto h_idx = 2;
    constexpr auto w_idx = 3;

    int64_t h_scale {
        output_shape->dim(h_idx).dim_value() /
        input_shape->dim(h_idx).dim_value()
    };
    int64_t w_scale {
        output_shape->dim(w_idx).dim_value() /
        input_shape->dim(w_idx).dim_value()
    };

    input_shape->mutable_dim(n_idx)->set_dim_value(batch);
    input_shape->mutable_dim(h_idx)->set_dim_value(block_h);
    input_shape->mutable_dim(w_idx)->set_dim_value(block_w);

    output_shape->mutable_dim(n_idx)->set_dim_value(batch);
    output_shape->mutable_dim(h_idx)->set_dim_value(block_h * h_scale);
    output_shape->mutable_dim(w_idx)->set_dim_value(block_w * w_scale);

    // remove shape info
    model.mutable_graph()->mutable_value_info()->Clear();
}


static int numPlanes(
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    int num_planes = 0;

    for (const auto & vi : vis) {
        num_planes += vi->format->numPlanes;
    }

    return num_planes;
}


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


[[nodiscard]]
static std::optional<std::string> checkIOInfo(
    const OrtTypeInfo * info
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    const OrtTensorTypeAndShapeInfo * tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(info, &tensor_info));

    ONNXTensorElementDataType element_type;
    checkError(ortapi->GetTensorElementType(tensor_info, &element_type));

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return set_error("expects network IO with type fp32");
    }

    size_t num_dims;
    checkError(ortapi->GetDimensionsCount(tensor_info, &num_dims));
    if (num_dims != 4) {
        return set_error("expects network with 4-D IO");
    }

    auto maybe_shape = getShape(tensor_info);
    if (std::holds_alternative<std::string>(maybe_shape)) {
        return set_error(std::get<std::string>(maybe_shape));
    }

    auto shape = std::get<std::array<int64_t, 4>>(maybe_shape);
    if (shape[0] != 1) {
        return set_error("batch size of network must be 1");
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkSession(
    const OrtSession * session
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    size_t num_inputs;
    checkError(ortapi->SessionGetInputCount(session, &num_inputs));

    if (num_inputs != 1) {
        return set_error("network input count must be 1, got " + std::to_string(num_inputs));
    }

    OrtTypeInfo * input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    if (auto err = checkIOInfo(input_type_info); err.has_value()) {
        return set_error(err.value());
    }

    ortapi->ReleaseTypeInfo(input_type_info);

    size_t num_outputs;
    checkError(ortapi->SessionGetOutputCount(session, &num_outputs));

    if (num_outputs != 1) {
        return "network output count must be 1, got " + std::to_string(num_outputs);
    }

    OrtTypeInfo * output_type_info;
    checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &output_type_info));

    if (auto err = checkIOInfo(output_type_info); err.has_value()) {
        return set_error(err.value());
    }

    ortapi->ReleaseTypeInfo(output_type_info);

    return {};
}

[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(
    const OrtSession * session,
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    OrtTypeInfo * input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    const OrtTensorTypeAndShapeInfo * input_tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));

    auto network_in_dims = std::get<std::array<int64_t, 4>>(getShape(input_tensor_info));

    int network_in_channels = static_cast<int>(network_in_dims[1]);
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return set_error("expects " + std::to_string(network_in_channels) + " input planes");
    }

    auto network_in_height = network_in_dims[2];
    auto network_in_width = network_in_dims[3];
    auto clip_in_height = vis.front()->height;
    auto clip_in_width = vis.front()->width;
    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return set_error("block size larger than clip dimension");
    }

    ortapi->ReleaseTypeInfo(input_type_info);

    return {};
}

static void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::array<int64_t, 4> & input_shape,
    const std::array<int64_t, 4> & output_shape
) noexcept {

    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];
}


struct vsOrtData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int pad;

    OrtEnv * environment;
    OrtSession * session;
    OrtValue * input_tensor;
    OrtValue * output_tensor;

    char * input_name;
    char * output_name;
};


static void VS_CC vsOrtInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsOrtGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(*instanceData);

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
        auto src_patch_shape = std::get<std::array<int64_t, 4>>(getShape(d->session, true));
        auto src_patch_h = src_patch_shape[2];
        auto src_patch_w = src_patch_shape[3];
        auto src_patch_w_bytes = src_patch_w * src_bytes;
        auto src_patch_bytes = src_patch_h * src_patch_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_patch_shape[1]);
        for (int i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_patch_w - 2 * d->pad;
        auto step_h = src_patch_h - 2 * d->pad;

        VSFrameRef * const dst_frame = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames.front(), core
        );
        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;
        auto dst_patch_shape = std::get<std::array<int64_t, 4>>(getShape(d->session, false));
        auto dst_patch_h = dst_patch_shape[2];
        auto dst_patch_w = dst_patch_shape[3];
        auto dst_patch_w_bytes = dst_patch_w * dst_bytes;
        auto dst_patch_bytes = dst_patch_h * dst_patch_w_bytes;
        auto dst_planes = dst_patch_shape[1];
        uint8_t * dst_ptrs[3] {};
        for (int i = 0; i < dst_planes; ++i) {
            dst_ptrs[i] = vsapi->getWritePtr(dst_frame, i);
        }

        auto h_scale = dst_patch_h / src_patch_h;
        auto w_scale = dst_patch_w / src_patch_w;

        const auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        int y = 0;
        while (true) {
            int y_pad_start = (y == 0) ? 0 : d->pad;
            int y_pad_end = (y == src_height - src_patch_h) ? 0 : d->pad;

            int x = 0;
            while (true) {
                int x_pad_start = (x == 0) ? 0 : d->pad;
                int x_pad_end = (x == src_width - src_patch_w) ? 0 : d->pad;

                {
                    uint8_t * input_buffer;
                    checkError(ortapi->GetTensorMutableData(
                        d->input_tensor,
                        reinterpret_cast<void **>(&input_buffer)
                    ));

                    for (const auto & _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * src_bytes
                        };

                        vs_bitblt(
                            input_buffer, src_patch_w_bytes,
                            src_ptr, src_stride,
                            src_patch_w_bytes, src_patch_h
                        );

                        input_buffer += src_patch_bytes;
                    }
                }

                checkError(ortapi->Run(
                    d->session,
                    nullptr,
                    &d->input_name, &d->input_tensor, 1,
                    &d->output_name, 1, &d->output_tensor
                ));

                {
                    uint8_t * output_buffer;
                    checkError(ortapi->GetTensorMutableData(
                        d->output_tensor,
                        reinterpret_cast<void **>(&output_buffer)
                    ));

                    for (int plane = 0; plane < dst_planes; ++plane) {
                        auto dst_ptr = (dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        );

                        vs_bitblt(
                            dst_ptr + (y_pad_start * dst_stride + x_pad_start * dst_bytes),
                            dst_stride,
                            output_buffer + (y_pad_start * dst_patch_w_bytes + x_pad_start * dst_bytes),
                            dst_patch_w_bytes,
                            dst_patch_w_bytes - (x_pad_start + x_pad_end) * dst_bytes,
                            dst_patch_h - (y_pad_start + y_pad_end)
                        );

                        output_buffer += dst_patch_bytes;
                    }
                }

                if (x + src_patch_w == src_width) {
                    break;
                }

                x = std::min(x + step_w, src_width - src_patch_w);
            }

            if (y + src_patch_h == src_height) {
                break;
            }

            y = std::min(y + step_h, src_height - src_patch_h);
        }

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsOrtFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d = static_cast<vsOrtData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    ortapi->ReleaseValue(d->output_tensor);
    ortapi->ReleaseValue(d->input_tensor);
    ortapi->ReleaseSession(d->session);
    ortapi->ReleaseEnv(d->environment);

    delete d;
}


static void VS_CC vsOrtCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<vsOrtData>() };

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

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    auto verbosity = static_cast<OrtLoggingLevel>(
        vsapi->propGetInt(in, "verbosity", 0, &error)
    );
    if (error) {
        verbosity = ORT_LOGGING_LEVEL_WARNING;
    }

    // match verbosity of vs-trt
    verbosity = static_cast<OrtLoggingLevel>(4 - static_cast<int>(verbosity));

    d->pad = int64ToIntS(vsapi->propGetInt(in, "pad", 0, &error));
    if (error) {
        d->pad = 0;
    }
    if (d->pad < 0) {
        return set_error("\"pad\" should be non-negative");
    }

    int error1, error2;
    size_t block_w = static_cast<size_t>(vsapi->propGetInt(in, "block_w", 0, &error1));
    size_t block_h = static_cast<size_t>(vsapi->propGetInt(in, "block_h", 0, &error2));
    if (!error1) { // manual specification triggered
        if (error2) {
            block_h = block_w;
        }
    } else {
        if (d->pad != 0) {
            return set_error("\"block_w\" must be specified");
        }

        // set block size to video dimensions
        block_w = in_vis.front()->width;
        block_h = in_vis.front()->height;
    }
    if (block_w - 2 * d->pad <= 0 || block_h - 2 * d->pad <= 0) {
        return set_error("\"pad\" too large");
    }

    const char * _provider = vsapi->propGetData(in, "provider", 0, &error);
    if (error) {
        _provider = "";
    }
    const std::string provider { _provider };

    {
        if (auto err = ortInit(); err.has_value()) {
            return set_error(err.value());
        }

        checkError(ortapi->CreateEnv(verbosity, "vs-ort", &d->environment));

        {
            OrtSessionOptions * session_options;
            checkError(ortapi->CreateSessionOptions(&session_options));
            checkError(ortapi->SetSessionExecutionMode(
                session_options,
                ExecutionMode::ORT_SEQUENTIAL
            ));
            checkError(ortapi->EnableMemPattern(session_options));

            // TODO: other providers
            if (provider == "CPU")
                ; // nothing to do
            else if (provider == "CUDA") {
#ifdef _MSC_VER
                    // Preload cuda dll from vsort directory.
                    static std::once_flag cuda_dll_preloaded_flag;
                    std::call_once(cuda_dll_preloaded_flag, []() {
                            extern void preloadCudaDlls();
                            preloadCudaDlls();
                    });
#endif

                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = device_id;

                #if ORT_API_VERSION >= 10
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // TODO: make an option
                #else
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::Exhaustive;
                #endif

                checkError(ortapi->SessionOptionsAppendExecutionProvider_CUDA(
                    session_options,
                    &cuda_options
                ));
            } else if (provider != "") {
                return set_error("unknwon provider " + provider);
            }

            std::ifstream onnx_stream(
                translateName(vsapi->propGetData(in, "network_path", 0, nullptr)),
                std::ios::in | std::ios::binary
            );

            if (!onnx_stream.good()) {
                return set_error("open .onnx failed");
            }

            std::string onnx_data {
                std::istreambuf_iterator<char>{ onnx_stream },
                std::istreambuf_iterator<char>{}
            };

            onnx::ModelProto onnx_proto;
            try {
                onnx::ParseProtoFromBytes(
                    &onnx_proto,
                    onnx_data.data(), std::size(onnx_data)
                );
            } catch (const std::runtime_error & e) {
                return set_error(e.what());
            }

            specifyShape(onnx_proto, block_w, block_h);

            onnx_data = onnx_proto.SerializeAsString();
            if (std::size(onnx_data) == 0) {
                return set_error("proto serialization failed");
            }

            checkError(ortapi->CreateSessionFromArray(
                d->environment,
                onnx_data.data(), std::size(onnx_data),
                session_options,
                &d->session
            ));

            ortapi->ReleaseSessionOptions(session_options);

            if (auto err = checkSession(d->session); err.has_value()) {
                return set_error(err.value());
            }
        }

        {
            OrtAllocator * allocator;
            checkError(ortapi->GetAllocatorWithDefaultOptions(&allocator));

            auto input_shape = std::get<std::array<int64_t, 4>>(getShape(d->session, true));

            checkError(ortapi->CreateTensorAsOrtValue(
                allocator,
                input_shape.data(),
                std::size(input_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &d->input_tensor
            ));

            checkError(ortapi->SessionGetInputName(d->session, 0, allocator, &d->input_name));


            auto output_shape = std::get<std::array<int64_t, 4>>(getShape(d->session, false));

            checkError(ortapi->CreateTensorAsOrtValue(
                allocator,
                output_shape.data(),
                std::size(output_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &d->output_tensor
            ));

            checkError(ortapi->SessionGetOutputName(d->session, 0, allocator, &d->output_name));

            if (auto err = checkNodesAndNetwork(d->session, in_vis); err.has_value()) {
                return set_error(err.value());
            }

            setDimensions(d->out_vi, input_shape, output_shape);
        }
    }

    vsapi->createFilter(
        in, out, "Model",
        vsOrtInit, vsOrtGetFrame, vsOrtFree,
        fmParallelRequests, 0, d.release(), core
    );
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin *plugin
) noexcept {

    configFunc(
        "io.github.amusementclub.vs_onnxruntime", "ort",
        "ONNX Runtime ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "network_path:data;"
        "pad:int:opt;"
        "block_w:int:opt;"
        "block_h:int:opt;"
        "provider:data:opt;" // "": Default (CPU), "CUDA": CUDA
        "device_id:int:opt;"
        "verbosity:int:opt;",
        vsOrtCreate,
        nullptr,
        plugin
    );
}
