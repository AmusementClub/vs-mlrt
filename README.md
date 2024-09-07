# vs-mlrt

This project provides VapourSynth ML filter runtimes for a variety of platforms:
 - x86 CPUs: [vsov-cpu](#vsov-openvino-based-pure-cpu--intel-gpu-runtime), [vsort-cpu](#vsort-onnx-runtime-based-cpugpu-runtime)
 - Intel GPU (both integrated & discrete): [vsov-gpu](#vsov-openvino-based-pure-cpu--intel-gpu-runtime), [vsncnn-vk](#vsncnn-ncnn-based-gpu-vulkan-runtime)
 - NVidia GPU: [vsort-cuda](#vsort-onnx-runtime-based-cpugpu-runtime), [vstrt](#vstrt-tensorrt-based-gpu-runtime), [vsncnn-vk](#vsncnn-ncnn-based-gpu-vulkan-runtime)
 - AMD GPU: [vsncnn-vk](#vsncnn-ncnn-based-gpu-vulkan-runtime), [vsmigx](#vsmigx-migraphx-based-gpu-runtime)
 - Apple SoC: [vsort-coreml](#vsort-onnx-runtime-based-cpugpu-runtime)

To simplify usage, we also provide a Python wrapper [vsmlrt.py](https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py)
for all bundled models and a unified interface to select different backends.

Please refer to [the wiki](https://github.com/AmusementClub/vs-mlrt/wiki) for supported models & usage information.

## vsov: OpenVINO-based Pure CPU & Intel GPU Runtime

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an AI inference runtime developed
by Intel, mainly targeting x86 CPUs and Intel GPUs.

The vs-openvino plugin provides optimized *pure* CPU & Intel GPU runtime for some popular AI filters.
Intel GPU supports Gen 8+ on Broadwell+ and the Arc series GPUs.

To install, download the latest release and extract them into your VS `plugins` directory.

Please visit the [vsov](vsov) directory for details.

## vsort: ONNX Runtime-based CPU/GPU Runtime

[ONNX Runtime](https://onnxruntime.ai/) is an AI inference runtime with many backends.

The vs-onnxruntime plugin provides optimized CPU and CUDA GPU runtime for some popular AI filters.

To install, download the latest release and extract them into your VS `plugins` directory.

Please visit the [vsort](vsort) directory for details.

## vstrt: TensorRT-based GPU Runtime

[TensorRT](https://developer.nvidia.com/tensorrt) is a highly optimized AI inference runtime
for NVidia GPUs. It uses benchmarking to find the optimal kernel to use for your specific
GPU, and so there is an extra step to build an engine from ONNX network on the machine
you are going to use the vstrt filter, and this extra step makes deploying models a little
harder than the other runtimes. However, the resulting performance is also typically
*much much better* than the CUDA backend of [vsort](vsort).

To install, download the latest release and extract them into your VS `plugins` directory.

Please visit the [vstrt](vstrt) directory for details.

## vsmigx: MIGraphX-based GPU Runtime

[MIGraphX](https://github.com/ROCm/AMDMIGraphX) is a highly optimized AI inference runtime
for AMD GPUs. It also uses benchmarking to find the optimal kernel, similar to vstrt.

To install, download the latest release and extract them into your VS `plugins` directory.

Please visit the [vsmigx](vsmigx) directory for details.

## vsncnn: NCNN-based GPU (Vulkan) Runtime

[ncnn](https://github.com/Tencent/ncnn) is a popular AI inference runtime. [vsncnn](vsncnn)
provides a vulkan based runtime for some AI filters. It includes support for on-the-fly
ONNX to ncnn native format conversion so as to provide a unified interface across all
runtimes provided by this project. As it uses the device-independent
[Vulkan](https://en.wikipedia.org/wiki/Vulkan) interface for GPU accelerated inference,
this plugin supports all GPUs that provides Vulkan interface (NVidia, AMD, Intel integrated &
discrete GPUs all provide this interface.) Another benefit is that it has a significant
smaller footprint than other GPU runtimes (both vsort and vstrt CUDA backends require >1GB
CUDA libraries.) The main drawback is that it's slower.

To install, download the latest release and extract them into your VS `plugins` directory.

Please visit the [vsncnn](vsncnn) directory for details.
