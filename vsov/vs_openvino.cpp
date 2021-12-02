#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <ie_core.hpp>

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
using namespace InferenceEngine;


static std::array<int, 4> getShape(
    const InferenceEngine::ExecutableNetwork & network, bool input
) noexcept {
    SizeVector dims;

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

static std::optional<std::string> specifyShape(
    InferenceEngine::CNNNetwork & network,
    size_t block_w,
    size_t block_h,
    size_t batch = 1
) noexcept {

    const auto & input_name = network.getInputsInfo().cbegin()->first;
    size_t channels = network.getInputShapes().cbegin()->second[1];

    ICNNNetwork::InputShapes shape {{
        input_name,
        { batch, channels, block_h, block_w }
    }};

    try {
        network.reshape(shape);
    } catch (const InferenceEngine::Exception& e) {
        return "specifyShape():"s + e.what();
    }

    return {};
}

static std::optional<std::string> checkNetwork(
    const ExecutableNetwork & network,
    const VSVideoInfo * vi,
    bool augmented
) noexcept {

    const auto & input_info = network.GetInputsInfo();
    if (auto num_inputs = std::size(input_info); num_inputs != 1) {
        return "input count must be 1, got " + std::to_string(num_inputs);
    }

    for (const auto & [_, info] : input_info) {
        if (info->getPrecision() != Precision::FP32) {
            return "expect clip with type fp32";
        }

        const auto & desc = info->getTensorDesc();
        if (desc.getLayout() != Layout::NCHW) {
            return "expect input with layout nchw";
        }
        const auto & dims = desc.getDims();
        if (dims.size() != 4) {
            return "expect 4-D input";
        }
        if (dims[0] != 1) {
            return "batch size must be 1";
        }
        if (dims[1] != 1 + augmented) {
            if (dims[1] != 3 + augmented) {
                return "unknown channels count " + std::to_string(dims[1]);
            }

            if (vi->format->numPlanes != 3) {
                return "clip does not have enough planes";
            }
            if (vi->format->subSamplingW || vi->format->subSamplingH) {
                return "clip must not be sub-sampled";
            }
        }

        if (dims[2] > vi->height || dims[3] > vi->width) {
            return "patch size larger than clip dimension";
        }
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
    VSNodeRef * node;
    std::unique_ptr<VSVideoInfo> out_vi;

    bool augmented;
    int pad;

    InferenceEngine::Core core;
    InferenceEngine::ExecutableNetwork executable_network;
    std::unordered_map<std::thread::id, InferenceEngine::InferRequest> infer_requests;

    std::string input_name;
    std::string output_name;

    float sigma;
};


static void VS_CC vsOvInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}


static const VSFrameRef *VS_CC vsOvGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const auto & src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto src_stride = vsapi->getStride(src_frame, 0);
        auto src_width = vsapi->getFrameWidth(src_frame, 0);
        auto src_height = vsapi->getFrameHeight(src_frame, 0);
        auto src_bytes = vsapi->getFrameFormat(src_frame)->bytesPerSample;
        auto src_patch_shape = getShape(d->executable_network, true);
        auto src_patch_h = src_patch_shape[2];
        auto src_patch_w = src_patch_shape[3];
        auto src_patch_w_bytes = src_patch_w * src_bytes;
        auto src_patch_bytes = src_patch_h * src_patch_w_bytes;
        auto src_planes = src_patch_shape[1] - int(d->augmented);
        const uint8_t * src_ptrs[3] {};
        for (int i = 0; i < src_planes; ++i) {
            src_ptrs[i] = vsapi->getReadPtr(src_frame, i);
        }

        auto step_w = src_patch_w - 2 * d->pad;
        auto step_h = src_patch_h - 2 * d->pad;

        VSFrameRef * const dst_frame = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frame, core
        );
        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getFrameFormat(dst_frame)->bytesPerSample;
        auto dst_patch_shape = getShape(d->executable_network, false);
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

        auto thread_id = std::this_thread::get_id();
        if (d->infer_requests.count(thread_id) == 0) {
            d->infer_requests.emplace(thread_id, d->executable_network.CreateInferRequest());
        }
        InferRequest & infer_request = d->infer_requests[thread_id];

        int y = 0;
        while (true) {
            int y_pad_start = (y == 0) ? 0 : d->pad;
            int y_pad_end = (y == src_height - src_patch_h) ? 0 : d->pad;

            int x = 0;
            while (true) {
                int x_pad_start = (x == 0) ? 0 : d->pad;
                int x_pad_end = (x == src_width - src_patch_w) ? 0 : d->pad;

                {
                    Blob::Ptr input = infer_request.GetBlob(d->input_name);

                    auto minput = as<MemoryBlob>(input);
                    auto minputHolder = minput->wmap();
                    uint8_t * input_buffer = minputHolder.as<uint8_t *>();

                    for (int plane = 0; plane < src_planes; ++plane) {
                        auto src_ptr = (src_ptrs[plane] +
                            y * src_stride + x * src_bytes
                        );

                        vs_bitblt(
                            input_buffer, src_patch_w_bytes,
                            src_ptr, src_stride,
                            src_patch_w_bytes, src_patch_h
                        );

                        input_buffer += src_patch_bytes;
                    }

                    if (d->augmented) {
                        std::vector<float> data(src_patch_bytes / sizeof(float), d->sigma);
                        memcpy(input_buffer, data.data(), src_patch_bytes);
                    }
                }

                infer_request.Infer();

                {
                    Blob::CPtr output = infer_request.GetBlob(d->output_name);

                    auto moutput = as<const MemoryBlob>(output);
                    auto moutputHolder = moutput->rmap();
                    const uint8_t * output_buffer = moutputHolder.as<const uint8_t *>();

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

        vsapi->freeFrame(src_frame);
        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsOvFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}


static void VS_CC vsOvCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<OVData>() };

    d->augmented = !!reinterpret_cast<intptr_t>(userData);

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->out_vi = std::make_unique<VSVideoInfo>(*vsapi->getVideoInfo(d->node)); // mutable

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, (__func__ + ": "s + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    auto in_vi = vsapi->getVideoInfo(d->node);
    if (in_vi->format->sampleType != stFloat || in_vi->format->bitsPerSample != 32) {
        return set_error("only floating point formats are supported");
    }

    int error;

    float sigma {};
    if (d->augmented) {
        sigma = static_cast<float>(vsapi->propGetFloat(in, "sigma", 0, &error));
        if (error) {
            sigma = 5.0f;
        }
        sigma /= 255.0f;
    }
    d->sigma = sigma;

    const char * device = vsapi->propGetData(in, "device", 0, &error);
    if (error) {
        device = "CPU";
    }

    d->pad = int64ToIntS(vsapi->propGetInt(in, "pad", 0, &error));
    if (error) {
        d->pad = 0;
    }

    if (d->pad < 0) {
        return set_error("\"pad\" should be non-negative");
    }

    {
        auto filename = translateName(vsapi->propGetData(in, "network_path", 0, nullptr));

        InferenceEngine::CNNNetwork network;
        try {
            network = d->core.ReadNetwork(filename);
        } catch (const InferenceEngine::Exception& e) {
            return set_error("ReadNetwork(): "s + e.what());
        } catch (const std::exception& e) {
            return set_error("Standard exception from compilation library: "s + e.what());
        }

        {
            int error1, error2;
            size_t block_w = static_cast<size_t>(vsapi->propGetInt(in, "block_w", 0, &error1));
            size_t block_h = static_cast<size_t>(vsapi->propGetInt(in, "block_h", 0, &error2));

            if (!error1) { // manual specification triggered
                if (error2) {
                    block_h = block_w;
                }
            } else {
                // set block size to video dimensions
                block_w = in_vi->width;
                block_h = in_vi->height;
            }

            if (auto err = specifyShape(network, block_w, block_h); err.has_value()) {
                return set_error(err.value());
            }
        }

        d->executable_network = d->core.LoadNetwork(network, device);

        if (auto err = checkNetwork(d->executable_network, in_vi, d->augmented); err.has_value()) {
            return set_error(err.value());
        }

        setDimensions(d->out_vi, d->executable_network);

        d->input_name = d->executable_network.GetInputsInfo().cbegin()->first;
        d->output_name = d->executable_network.GetOutputsInfo().cbegin()->first;

        VSCoreInfo core_info;
        vsapi->getCoreInfo2(core, &core_info);
        d->infer_requests.reserve(core_info.numThreads);
    }

    if (d->augmented) {
        vsapi->createFilter(
            in, out, "AugmentedModel",
            vsOvInit, vsOvGetFrame, vsOvFree,
            fmParallel, 0, d.release(), core
        );
    } else {
        vsapi->createFilter(
            in, out, "Model",
            vsOvInit, vsOvGetFrame, vsOvFree,
            fmParallel, 0, d.release(), core
        );
    }
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) noexcept {

    configFunc(
        "com.amusementclub.vs_openvino", "ov", "OpenVINO wrapper",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("Model",
        "clip:clip;"
        "network_path:data;"
        "device:data:opt;" // "CPU": CPU
        "pad:int:opt;"
        "block_w:int:opt;"
        "block_h:int:opt;",
        vsOvCreate,
        reinterpret_cast<void *>(intptr_t(false)), // not augmented
        plugin
    );

    registerFunc("AugmentedModel",
        "clip:clip;"
        "network_path:data;"
        "sigma:float:opt;"
        "device:data:opt;"
        "pad:int:opt;"
        "block_w:int:opt;"
        "block_h:int:opt;",
        vsOvCreate,
        reinterpret_cast<void *>(intptr_t(true)), // augmented
        plugin
    );
}
