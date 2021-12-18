# vs-mlrt

VapourSynth ML filter runtimes.

Please see [the wiki](https://github.com/AmusementClub/vs-mlrt/wiki) for supported models.

## vsov: OpenVINO-based Pure CPU Runtime

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an AI inference runtime developed
by Intel, mainly targeting x86 CPUs and Intel GPUs.

The vs-openvino plugin provides optimized *pure* CPU runtime for some popular AI filters,
with Intel GPU support planned in the future.

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
