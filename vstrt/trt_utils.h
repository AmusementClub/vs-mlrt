#ifndef VSTRT_TRT_UTILS_H_
#define VSTRT_TRT_UTILS_H_

#include <cstdint>
#include <memory>
#include <iostream>
#include <optional>
#include <string>
#include <variant>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include "cuda_helper.h"
#include "cuda_utils.h"

using ErrorMessage = std::string;

struct RequestedTileSize {
    int tile_w;
    int tile_h;
};

struct VideoSize {
    int width;
    int height;
};

using TileSize = std::variant<RequestedTileSize, VideoSize>;

struct InferenceInstance {
    MemoryResource src;
    MemoryResource dst;
    StreamResource stream;
    std::unique_ptr<nvinfer1::IExecutionContext> exec_context;
    GraphExecResource graphexec;
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= verbosity) {
            std::cerr << message << '\n';
        }
    }

public:
    Logger() = default;

    void set_verbosity(Severity verbosity) noexcept {
        this->verbosity = verbosity;
    }

private:
    Severity verbosity;
};

static inline
std::optional<int> selectProfile(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const TileSize & tile_size,
    int batch_size = 1
) noexcept {

    int tile_w, tile_h;
    if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        tile_w = std::get<RequestedTileSize>(tile_size).tile_w;
        tile_h = std::get<RequestedTileSize>(tile_size).tile_h;
    } else {
        tile_w = std::get<VideoSize>(tile_size).width;
        tile_h = std::get<VideoSize>(tile_size).height;
    }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = engine->getIOTensorName(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    // finds the optimal profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims opt_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kOPT
        );
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims opt_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kOPT
        );
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        if (opt_dims.d[0] != batch_size) {
            continue;
        }
        if (opt_dims.d[2] == tile_h && opt_dims.d[3] == tile_w) {
            return i;
        }
    }

    // finds the first eligible profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims min_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMIN
        );
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims min_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMIN
        );
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        if (min_dims.d[0] > batch_size) {
            continue;
        }
        if (min_dims.d[2] > tile_h || min_dims.d[3] > tile_w) {
            continue;
        }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims max_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMAX
        );
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims max_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMAX
        );
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        if (max_dims.d[0] < batch_size) {
            continue;
        }
        if (max_dims.d[2] < tile_h || max_dims.d[3] < tile_w) {
            continue;
        }

        return i;
    }

    // returns not-found
    return {};
}

static inline
std::optional<ErrorMessage> enqueue(
    const MemoryResource & src,
    const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    checkError(cudaMemcpyAsync(
        src.d_data, src.h_data, src.size,
        cudaMemcpyHostToDevice, stream
    ));

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);

    if (!exec_context->setTensorAddress(input_name, src.d_data.data)) {
        return set_error("set input tensor address failed");
    }
    if (!exec_context->setTensorAddress(output_name, dst.d_data.data)) {
        return set_error("set output tensor address failed");
    }
    if (!exec_context->enqueueV3(stream)) {
        return set_error("enqueue error");
    }
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    void * bindings[] {
        static_cast<void *>(src.d_data.data),
        static_cast<void *>(dst.d_data.data)
    };

    if (!exec_context->enqueueV2(bindings, stream, nullptr)) {
        return set_error("enqueue error");
    }
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    checkError(cudaMemcpyAsync(
        dst.h_data, dst.d_data, dst.size,
        cudaMemcpyDeviceToHost, stream
    ));

    return {};
}

static inline
std::variant<ErrorMessage, GraphExecResource> getGraphExec(
    const MemoryResource & src, const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    // flush deferred internal state update
    // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/developer-guide/index.html#cuda-graphs
    {
        auto result = enqueue(src, dst, exec_context, stream);
        if (result.has_value()) {
            return set_error(result.value());
        }
        checkError(cudaStreamSynchronize(stream));
    }

    checkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    {
        auto result = enqueue(src, dst, exec_context, stream);
        if (result.has_value()) {
            return set_error(result.value());
        }
    }
    cudaGraph_t graph;
    checkError(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphexec;
    checkError(cudaGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
    checkError(cudaGraphDestroy(graph));

    return graphexec;
}

static inline
size_t getSize(
    const nvinfer1::Dims & dim
) noexcept {

    size_t ret = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
        ret *= dim.d[i];
    }
    return ret;
}

static inline
size_t getBytesPerSample(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
#if (NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 861
        case nvinfer1::DataType::kFP8:
            return 1;
#endif // (NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 861
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kINT64:
            return 8;
#endif // NV_TENSORRT_MAJOR >= 9
        default:
            return 0;
    }
}

static inline
std::variant<ErrorMessage, InferenceInstance> getInstance(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const std::optional<int> & profile_index,
    const TileSize & tile_size,
    bool use_cuda_graph,
    bool & is_dynamic
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    StreamResource stream {};
    checkError(cudaStreamCreateWithFlags(&stream.data, cudaStreamNonBlocking));

    auto exec_context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext()
    );

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    if (!exec_context->allInputDimensionsSpecified()) {
        if (!profile_index.has_value()) {
            return set_error("no valid optimization profile found");
        }

        is_dynamic = true;

        exec_context->setOptimizationProfileAsync(profile_index.value(), stream);
        checkError(cudaStreamSynchronize(stream));

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims dims = exec_context->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        dims.d[0] = 1;

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            dims.d[2] = std::get<RequestedTileSize>(tile_size).tile_h;
            dims.d[3] = std::get<RequestedTileSize>(tile_size).tile_w;
        } else {
            dims.d[2] = std::get<VideoSize>(tile_size).height;
            dims.d[3] = std::get<VideoSize>(tile_size).width;
        }
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        exec_context->setInputShape(input_name, dims);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        exec_context->setBindingDimensions(0, dims);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    } else if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        is_dynamic = false;

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        nvinfer1::Dims dims = exec_context->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            if (dims.d[2] != std::get<RequestedTileSize>(tile_size).tile_h ||
                dims.d[3] != std::get<RequestedTileSize>(tile_size).tile_w
            ) {
                return set_error("requested tile size not applicable");
            }
        } else {
            if (dims.d[2] != std::get<VideoSize>(tile_size).height ||
                dims.d[3] != std::get<VideoSize>(tile_size).width
            ) {
                return set_error("not supported video dimensions");
            }
        }
    }

    MemoryResource src {};
    {
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto dim = exec_context->getTensorShape(input_name);
        auto type = engine->getTensorDataType(input_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto dim = exec_context->getBindingDimensions(0);
        auto type = engine->getBindingDataType(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        auto size = getSize(dim) * getBytesPerSample(type);

        Resource<uint8_t *, cudaFree> d_data {};
        checkError(cudaMalloc(&d_data.data, size));

        Resource<uint8_t *, cudaFreeHost> h_data {};
        checkError(cudaMallocHost(&h_data.data, size, cudaHostAllocWriteCombined));

        src = MemoryResource{
            .h_data = std::move(h_data),
            .d_data = std::move(d_data),
            .size=size
        };
    }

    MemoryResource dst {};
    {
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto dim = exec_context->getTensorShape(output_name);
        auto type = engine->getTensorDataType(output_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        auto dim = exec_context->getBindingDimensions(1);
        auto type = engine->getBindingDataType(1);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

        auto size = getSize(dim) * getBytesPerSample(type);

        Resource<uint8_t *, cudaFree> d_data {};
        checkError(cudaMalloc(&d_data.data, size));

        Resource<uint8_t *, cudaFreeHost> h_data {};
        checkError(cudaMallocHost(&h_data.data, size));

        dst = MemoryResource{
            .h_data = std::move(h_data),
            .d_data = std::move(d_data),
            .size=size
        };
    }

    GraphExecResource graphexec {};
    if (use_cuda_graph) {
        auto result = getGraphExec(
            src, dst,
            exec_context, stream
        );
        if (std::holds_alternative<GraphExecResource>(result)) {
            graphexec = std::move(std::get<GraphExecResource>(result));
        } else {
            return set_error(std::get<ErrorMessage>(result));
        }
    }

    return InferenceInstance{
        .src = std::move(src),
        .dst = std::move(dst),
        .stream = std::move(stream),
        .exec_context = std::move(exec_context),
        .graphexec = std::move(graphexec)
    };
}

static inline
std::optional<ErrorMessage> checkEngine(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine
) noexcept {

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    int num_bindings = engine->getNbIOTensors();
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    int num_bindings = engine->getNbBindings();
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    if (num_bindings != 2) {
        return "network binding count must be 2, got " + std::to_string(num_bindings);
    }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    if (engine->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT) {
        return "the first binding should be an input binding";
    }
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    if (!engine->bindingIsInput(0)) {
        return "the first binding should be an input binding";
    }
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    const nvinfer1::Dims & input_dims = engine->getTensorShape(input_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    const nvinfer1::Dims & input_dims = engine->getBindingDimensions(0);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    if (input_dims.nbDims != 4) {
        return "expects network with 4-D input";
    }
    if (input_dims.d[0] != 1) {
        return "batch size of network input must be 1";
    }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    if (engine->getTensorIOMode(output_name) != nvinfer1::TensorIOMode::kOUTPUT) {
        return "the second binding should be an output binding";
    }
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    if (engine->bindingIsInput(1)) {
        return "the second binding should be an output binding";
    }
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    const nvinfer1::Dims & output_dims = engine->getTensorShape(output_name);
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    const nvinfer1::Dims & output_dims = engine->getBindingDimensions(1);
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

    if (output_dims.nbDims != 4) {
        return "expects network with 4-D output";
    }
    if (output_dims.d[0] != 1) {
        return "batch size of network output must be 1";
    }

    int out_channels = output_dims.d[1];
    if (out_channels != 1 && out_channels != 3) {
        return "output dimensions must be 1 or 3";
    }

    int in_height = input_dims.d[2];
    int in_width = input_dims.d[3];
    int out_height = output_dims.d[2];
    int out_width = output_dims.d[3];
    if (out_height % in_height != 0 || out_width % in_width != 0) {
        return "output dimensions must be divisible by input dimensions";
    }

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    for (const auto & name : { input_name, output_name }) {
        if (engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::string{ name } + " should reside on device";
        }
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    for (int i = 0; i < 2; i++) {
        if (engine->getLocation(i) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::to_string(i) + " should reside on device";
        }
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        if (engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
        if (engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
#endif // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    }

    return {};
}

static inline
std::variant<ErrorMessage, std::unique_ptr<nvinfer1::ICudaEngine>> initEngine(
    const char * engine_data, size_t engine_nbytes,
    const std::unique_ptr<nvinfer1::IRuntime> & runtime
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    std::unique_ptr<nvinfer1::ICudaEngine> engine {
        runtime->deserializeCudaEngine(engine_data, engine_nbytes)
    };

    if (!engine) {
        return set_error("engine deserialization failed");
    }

    if (auto err = checkEngine(engine); err.has_value()) {
        return set_error(err.value());
    }

    return engine;
}

// 0: integer, 1: float
static inline
int getSampleType(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kHALF:
#if (NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 861
        case nvinfer1::DataType::kFP8:
#endif // (NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR) * 10 + NV_TENSORRT_PATCH >= 861
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kBF16:
#endif // NV_TENSORRT_MAJOR >= 9
            return 1;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kINT64:
#endif // NV_TENSORRT_MAJOR >= 9
            return 0;
        default:
            return -1;
    }
}

#endif // VSTRT_TRT_UTILS_H_
