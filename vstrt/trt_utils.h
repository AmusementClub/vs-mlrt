#ifndef VSTRT_TRT_UTILS_H_
#define VSTRT_TRT_UTILS_H_

#include <cstdint>
#include <memory>
#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

    // finds the optimal profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
        nvinfer1::Dims opt_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kOPT
        );
        if (opt_dims.d[0] != batch_size) {
            continue;
        }
        if (opt_dims.d[2] == tile_h && opt_dims.d[3] == tile_w) {
            return i;
        }
    }

    // finds the first eligible profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
        nvinfer1::Dims min_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMIN
        );
        if (min_dims.d[0] > batch_size) {
            continue;
        }
        if (min_dims.d[2] > tile_h || min_dims.d[3] > tile_w) {
            continue;
        }

        nvinfer1::Dims max_dims = engine->getProfileDimensions(
            0, i, nvinfer1::OptProfileSelector::kMAX
        );
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

    void * bindings[] {
        static_cast<void *>(src.d_data.data),
        static_cast<void *>(dst.d_data.data)
    };

    if (!exec_context->enqueueV2(bindings, stream, nullptr)) {
        return set_error("enqueue error");
    }

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

    if (!exec_context->allInputDimensionsSpecified()) {
        if (!profile_index.has_value()) {
            return set_error("no valid optimization profile found");
        }

        is_dynamic = true;

        exec_context->setOptimizationProfileAsync(profile_index.value(), stream);
        checkError(cudaStreamSynchronize(stream));

        nvinfer1::Dims dims = exec_context->getBindingDimensions(0);
        dims.d[0] = 1;

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            dims.d[2] = std::get<RequestedTileSize>(tile_size).tile_h;
            dims.d[3] = std::get<RequestedTileSize>(tile_size).tile_w;
        } else {
            dims.d[2] = std::get<VideoSize>(tile_size).height;
            dims.d[3] = std::get<VideoSize>(tile_size).width;
        }
        exec_context->setBindingDimensions(0, dims);
    } else if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        return set_error("Engine has no dynamic dimensions");
    }

    MemoryResource src {};
    {
        auto dim = exec_context->getBindingDimensions(0);
        auto size = getSize(dim) * sizeof(float);

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
        auto dim = exec_context->getBindingDimensions(1);
        auto size = getSize(dim) * sizeof(float);\

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

    int num_bindings = engine->getNbBindings();

    if (num_bindings != 2) {
        return "network binding count must be 2, got " + std::to_string(num_bindings);
    }

    if (!engine->bindingIsInput(0)) {
        return "the first binding should be an input binding";
    }
    const nvinfer1::Dims & input_dims = engine->getBindingDimensions(0);
    if (input_dims.nbDims != 4) {
        return "expects network with 4-D input";
    }
    if (input_dims.d[0] != 1) {
        return "batch size of network input must be 1";
    }

    if (engine->bindingIsInput(1)) {
        return "the second binding should be an output binding";
    }
    const nvinfer1::Dims & output_dims = engine->getBindingDimensions(1);
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

    for (int i = 0; i < num_bindings; ++i) {
        if (engine->getLocation(i) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::to_string(i) + " should reside on device";
        }

        if (engine->getBindingDataType(i) != nvinfer1::DataType::kFLOAT) {
            return "expects network IO with type fp32";
        }

        if (engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
    }

    return {};
}

static inline
std::variant<ErrorMessage, std::unique_ptr<nvinfer1::ICudaEngine>> initEngine(
    const std::vector<char> & engine_data,
    const std::unique_ptr<nvinfer1::IRuntime> & runtime
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    std::unique_ptr<nvinfer1::ICudaEngine> engine {
        runtime->deserializeCudaEngine(engine_data.data(), std::size(engine_data))
    };

    if (!engine) {
        return set_error("engine deserialization failed");
    }

    if (auto err = checkEngine(engine); err.has_value()) {
        return set_error(err.value());
    }

    return engine;
}

#endif // VSTRT_TRT_UTILS_H_
