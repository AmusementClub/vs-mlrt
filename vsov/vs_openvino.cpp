#include <array>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <onnx/common/version.h>
#include <onnx/proto_utils.h>
#include <onnx/shape_inference/implementation.h>
#include <ie_core.hpp>

#include "config.h"

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

using namespace std::string_literals;

static const VSPlugin * myself = nullptr;

static std::array<int, 4> getShape(
    const InferenceEngine::ExecutableNetwork & network,
    bool input
) noexcept {

    InferenceEngine::SizeVector dims;

    if (input) {
        dims = network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims();
    } else {
        dims = network.GetOutputsInfo().cbegin()->second->getTensorDesc().getDims();
    }

    std::array<int, 4> ret;
    for (int i = 0; i < std::size(ret); ++i) {
        ret[i] = static_cast<int>(dims[i]);
    }

    return ret;
}


[[nodiscard]]
static std::optional<std::string> specifyShape(
    ONNX_NAMESPACE::ModelProto & model,
    int64_t tile_w,
    int64_t tile_h,
    int64_t batch = 1
) noexcept {

    ONNX_NAMESPACE::TensorShapeProto * input_shape {
        model
            .mutable_graph()
            ->mutable_input(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };
    ONNX_NAMESPACE::TensorShapeProto * output_shape {
        model
            .mutable_graph()
            ->mutable_output(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };

    constexpr auto n_idx = 0;
    constexpr auto h_idx = 2;
    constexpr auto w_idx = 3;

    input_shape->mutable_dim(n_idx)->set_dim_value(batch);
    input_shape->mutable_dim(h_idx)->set_dim_value(tile_h);
    input_shape->mutable_dim(w_idx)->set_dim_value(tile_w);

    output_shape->mutable_dim(n_idx)->set_dim_value(batch);
    output_shape->mutable_dim(h_idx)->clear_dim_value();
    output_shape->mutable_dim(w_idx)->clear_dim_value();

    // remove shape info
    model.mutable_graph()->mutable_value_info()->Clear();

    try {
        ONNX_NAMESPACE::shape_inference::InferShapes(model);
    } catch (const ONNX_NAMESPACE::InferenceError & e) {
        return e.what();
    }

    return {};
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


template <typename T>
[[nodiscard]]
static std::optional<std::string> checkIOInfo(
    const T & info
) noexcept {

    if (info->getPrecision() != InferenceEngine::Precision::FP32) {
        return "expects network IO with type fp32";
    }
    const auto & desc = info->getTensorDesc();
    if (desc.getLayout() != InferenceEngine::Layout::NCHW) {
        return "expects network IO with layout NCHW";
    }
    const auto & dims = desc.getDims();
    if (dims.size() != 4) {
        return "expects network with 4-D IO";
    }
    // 0: dynamic onnx model is loaded with empty dimensions
    if (dims[0] != 1 && dims[0] != 0) {
        return "batch size of network must be 1";
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNetwork(
    const InferenceEngine::CNNNetwork & network
) noexcept {

    const auto & inputs_info = network.getInputsInfo();

    if (auto num_inputs = std::size(inputs_info); num_inputs != 1) {
        return "network input count must be 1, got " + std::to_string(num_inputs);
    }

    const auto & input_info = inputs_info.cbegin()->second;
    if (auto err = checkIOInfo(input_info); err.has_value()) {
        return err.value();
    }

    const auto & outputs_info = network.getOutputsInfo();

    if (auto num_outputs = std::size(outputs_info); num_outputs != 1) {
        return "network output count must be 1, got " + std::to_string(num_outputs);
    }

    const auto & output_info = outputs_info.cbegin()->second;
    if (auto err = checkIOInfo(output_info); err.has_value()) {
        return err.value();
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(
    const InferenceEngine::ExecutableNetwork & network,
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    const auto & network_in_dims = (
        network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims()
    );

    int network_in_channels = static_cast<int>(network_in_dims[1]);
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return "expects " + std::to_string(network_in_channels) + " input planes";
    }

    auto network_in_height = network_in_dims[2];
    auto network_in_width = network_in_dims[3];
    auto clip_in_height = vis.front()->height;
    auto clip_in_width = vis.front()->width;
    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return "tile size larger than clip dimension";
    }

    return {};
}


static void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const InferenceEngine::ExecutableNetwork & network
) noexcept {

    auto in_dims = network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims();
    auto out_dims = network.GetOutputsInfo().cbegin()->second->getTensorDesc().getDims();

    vi->height *= out_dims[2] / in_dims[2];
    vi->width *= out_dims[3] / in_dims[3];
}


struct OVData {
    std::vector<VSNodeRef *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int overlap_w, overlap_h;

    InferenceEngine::Core core;
    InferenceEngine::ExecutableNetwork executable_network;
    std::unordered_map<std::thread::id, InferenceEngine::InferRequest> infer_requests;

    std::string input_name;
    std::string output_name;
};


static void VS_CC vsOvInit(
    VSMap *in,
    VSMap *out,
    void **instanceData,
    VSNode *node,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsOvGetFrame(
    int n,
    int activationReason,
    void **instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(*instanceData);

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
        auto src_tile_shape = getShape(d->executable_network, true);
        auto src_tile_h = src_tile_shape[2];
        auto src_tile_w = src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;
        auto src_tile_bytes = src_tile_h * src_tile_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_tile_shape[1]);
        for (int i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format->numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        VSFrameRef * const dst_frame = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames.front(), core
        );
        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;
        auto dst_tile_shape = getShape(d->executable_network, false);
        auto dst_tile_h = dst_tile_shape[2];
        auto dst_tile_w = dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_tile_bytes = dst_tile_h * dst_tile_w_bytes;
        auto dst_planes = dst_tile_shape[1];
        std::array<uint8_t *, 3> dst_ptrs {};
        for (int i = 0; i < dst_planes; ++i) {
            dst_ptrs[i] = vsapi->getWritePtr(dst_frame, i);
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        auto thread_id = std::this_thread::get_id();
        if (d->infer_requests.count(thread_id) == 0) {
            d->infer_requests.emplace(thread_id, d->executable_network.CreateInferRequest());
        }
        InferenceEngine::InferRequest & infer_request = d->infer_requests[thread_id];

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
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    InferenceEngine::Blob::Ptr input = infer_request.GetBlob(d->input_name);

                    auto minput = input->as<InferenceEngine::MemoryBlob>();
                    auto minputHolder = minput->wmap();
                    uint8_t * input_buffer = minputHolder.as<uint8_t *>();

                    for (const auto & _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * src_bytes
                        };

                        vs_bitblt(
                            input_buffer, src_tile_w_bytes,
                            src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h
                        );

                        input_buffer += src_tile_bytes;
                    }
                }

                try {
                    infer_request.Infer();
                } catch (const InferenceEngine::Exception & e) {
                    return set_error(e.what());
                }

                {
                    InferenceEngine::Blob::CPtr output = infer_request.GetBlob(d->output_name);

                    auto moutput = output->as<const InferenceEngine::MemoryBlob>();
                    auto moutputHolder = moutput->rmap();
                    const uint8_t * output_buffer = moutputHolder.as<const uint8_t *>();

                    for (int plane = 0; plane < dst_planes; ++plane) {
                        uint8_t * dst_ptr = (dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        );

                        vs_bitblt(
                            dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                            dst_stride,
                            output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                            dst_tile_h - (y_crop_start + y_crop_end)
                        );

                        output_buffer += dst_tile_bytes;
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

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsOvFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    delete d;
}


static void VS_CC vsOvCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<OVData>() };

    int num_nodes = vsapi->propNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->propGetNode(in, "clips", i, nullptr));
    }

    const auto set_error = [&](const std::string & error_message) {
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

    const char * device = vsapi->propGetData(in, "device", 0, &error);
    if (error) {
        device = "CPU";
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

    std::string path { vsapi->propGetData(in, "network_path", 0, nullptr) };
    bool builtin = !!vsapi->propGetInt(in, "builtin", 0, &error);
    if (builtin) {
        const char *modeldir = vsapi->propGetData(in, "builtindir", 0, &error);
        if (!modeldir) modeldir = "models";
        path = std::string(modeldir) + "/" + path;
        std::string dir { vsapi->getPluginPath(myself) };
        dir = dir.substr(0, dir.rfind('/') + 1);
        path = dir + path;
    }

    std::string onnx_data;
    {
        std::ifstream onnx_stream(
            translateName(path.c_str()),
            std::ios::binary | std::ios::ate
        );

        if (!onnx_stream.good()) {
            return set_error("open "s + path + " failed"s);
        }

        onnx_data.resize(onnx_stream.tellg());
        onnx_stream.seekg(0, std::ios::beg);
        onnx_stream.read(onnx_data.data(), onnx_data.size());
    }

    ONNX_NAMESPACE::ModelProto onnx_proto;
    try {
        ONNX_NAMESPACE::ParseProtoFromBytes(
            &onnx_proto,
            onnx_data.data(), std::size(onnx_data)
        );
    } catch (const std::runtime_error & e) {
        return set_error(e.what());
    }

    if (auto err = specifyShape(onnx_proto, tile_w, tile_h); err.has_value()) {
        return set_error(err.value());
    }

    onnx_data = onnx_proto.SerializeAsString();
    if (std::size(onnx_data) == 0) {
        return set_error("proto serialization failed");
    }

    {
        InferenceEngine::CNNNetwork network;
        try {
            auto empty = InferenceEngine::Blob::CPtr();
            network = d->core.ReadNetwork(onnx_data, empty);
        } catch (const InferenceEngine::Exception& e) {
            return set_error("ReadNetwork(): "s + e.what());
        } catch (const std::exception& e) {
            return set_error("Standard exception from compilation library: "s + e.what());
        }

        if (auto err = checkNetwork(network); err.has_value()) {
            return set_error(err.value());
        }

        d->executable_network = d->core.LoadNetwork(network, device);

        if (auto err = checkNodesAndNetwork(d->executable_network, in_vis); err.has_value()) {
            return set_error(err.value());
        }

        setDimensions(d->out_vi, d->executable_network);

        d->input_name = d->executable_network.GetInputsInfo().cbegin()->first;
        d->output_name = d->executable_network.GetOutputsInfo().cbegin()->first;

        VSCoreInfo core_info;
        vsapi->getCoreInfo2(core, &core_info);
        d->infer_requests.reserve(core_info.numThreads);
    }

    vsapi->createFilter(
        in, out, "Model",
        vsOvInit, vsOvGetFrame, vsOvFree,
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
        "io.github.amusementclub.vs_openvino", "ov", "OpenVINO ML Filter Runtime",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clips:clip[];"
        "network_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device:data:opt;" // "CPU": CPU
        "builtin:int:opt;"
        "builtindir:data:opt;"
        , vsOvCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        std::ostringstream ostream;
        ostream << IE_VERSION_MAJOR << '.' << IE_VERSION_MINOR << '.' << IE_VERSION_PATCH;
        vsapi->propSetData(out, "inference_engine_version", ostream.str().c_str(), -1, paReplace);

        vsapi->propSetData(
            out, "onnx_version",
            ONNX_NAMESPACE::LAST_RELEASE_VERSION, -1, paReplace
        );

        vsapi->propSetData(out, "path", vsapi->getPluginPath(myself), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
