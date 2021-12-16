#ifndef VSTRT_UTILS_H_
#define VSTRT_UTILS_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <NvInferRuntime.h>

#include <VapourSynth.h>

static inline
void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context
) noexcept {

    const nvinfer1::Dims & in_dims = exec_context->getBindingDimensions(0);
    int in_height = in_dims.d[2];
    int in_width = in_dims.d[3];

    const nvinfer1::Dims & out_dims = exec_context->getBindingDimensions(1);
    int out_height = out_dims.d[2];
    int out_width = out_dims.d[3];

    vi->height *= out_height / in_height;
    vi->width *= out_width / in_width;
}

static inline
std::vector<const VSVideoInfo *> getVideoInfo(
    const VSAPI * vsapi,
    const std::vector<VSNodeRef *> & nodes
) noexcept {

    std::vector<const VSVideoInfo *> vis;
    vis.reserve(std::size(nodes));

    for (const auto & node : nodes) {
        vis.emplace_back(vsapi->getVideoInfo(node));
    }
  
    return vis;
}

static inline
std::vector<const VSFrameRef *> getFrames(
    int n,
    const VSAPI * vsapi,
    VSFrameContext * frameCtx,
    const std::vector<VSNodeRef *> & nodes
) noexcept {

    std::vector<const VSFrameRef *> frames;
    frames.reserve(std::size(nodes));

    for (const auto & node : nodes) {
        frames.emplace_back(vsapi->getFrameFilter(n, node, frameCtx));
    }

    return frames;
}

static inline
std::optional<std::string> checkNodes(
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

static inline
std::optional<std::string> checkNodesAndContext(
    const std::unique_ptr<nvinfer1::IExecutionContext> & execution_context,
    const std::vector<const VSVideoInfo *> & vis
) noexcept {

    const nvinfer1::Dims & network_in_dims = execution_context->getBindingDimensions(0);

    int network_in_channels = network_in_dims.d[1];
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return "expects " + std::to_string(network_in_channels) + " input planes";
    }

    int network_in_height = network_in_dims.d[2];
    int network_in_width = network_in_dims.d[3];
    int clip_in_height = vis[0]->height;
    int clip_in_width = vis[0]->width;

    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return "tile size larger than clip dimension";
    }

    return {};
}

#endif // VSTRT_UTILS_H_
