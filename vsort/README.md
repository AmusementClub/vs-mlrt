# VapourSynth ONNX Runtime

The vs-onnxruntime plugin provides optimized CPU & CUDA runtime for some popular AI filters.

## Building and Installation

To build, you will need [ONNX Runtime](https://www.onnxruntime.ai/), [protobuf](https://github.com/protocolbuffers/protobuf), [ONNX](https://github.com/onnx/onnx) and their dependencies.

Please refer to [ONNX Runtime Docs](https://onnxruntime.ai/docs/install/) for installation notes.
Or, you can use our prebuilt Windows binary releases from [AmusementClub](https://github.com/AmusementClub/onnxruntime/releases/latest/).

Please refer to our [github actions workflow](../.github/workflows/windows-ort.yml) for sample building instructions.

If you only use the CPU backend, then you just need to extract binary release into your `vapoursynth/plugins` directory.

However, if you also use the CUDA backend, you will need to download some CUDA libraries as well, please see the release page for details. Those CUDA libraries also need to be extracted into VS `vapoursynth/plugins` directory. The plugin will try to load them from `vapoursynth/plugins/vsort/` directory or `vapoursynth/plugins/vsmlrt-cuda/` directory.

## Usage

Prototype: `core.ort.Model(clip[] clips, string network_path[, int[] overlap = None, int[] tilesize = None, string provider = "", int device_id = 0, int verbosity = 2, bint cudnn_benchmark = True, bint builtin = False, string builtindir="models", bint fp16 = False, bint path_is_serialization = False, bint use_cuda_graph = False])`

Arguments:
 - `clip[] clips`: the input clips, only 32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
 - `string network_path`: the path to the network in ONNX format.
 - `int[] overlap`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `overlap` argument specifies the overlapping (horizontal and vertical, or both, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended overlapping size.
 - `int[] tilesize`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the tile size (horizontal and vertical, or both, including the overlapping). Please refer to network specific docs on the recommended tile size.
 - `string provider`: Specifies the device to run the inference on.
   - `"CPU"` or `""`: pure CPU backend
   - `"CUDA"`: CUDA GPU backend, requires Nvidia Maxwell+ GPUs.
   - `"DML"`: DirectML backend
   - `"COREML"`: CoreML backend
 - `int device_id`: select the GPU device for the CUDA backend.'
 - `int verbosity`: specify the verbosity of logging, the default is warning.
   - 0: fatal error only, `ORT_LOGGING_LEVEL_FATAL`
   - 1: also errors, `ORT_LOGGING_LEVEL_ERROR`
   - 2: also warnings, `ORT_LOGGING_LEVEL_WARNING`
   - 3: also info, `ORT_LOGGING_LEVEL_INFO`
   - 4: everything, `ORT_LOGGING_LEVEL_VERBOSE`
 - `bint cudnn_benchmark`: whether to let cuDNN use benchmarking to search for the best convolution kernel to use. Default True. It might incur some startup latency.
 - `bint builtin`: whether to load the model from the VS plugins directory, see also `builtindir`.
 - `string builtindir`: the model directory under VS plugins directory for builtin models, default "models".
 - `bint fp16`: whether to quantize model to fp16 for faster and memory efficient computation.
 - `bint path_is_serialization`: whether the `network_path` argument specifies an onnx serialization of type `bytes`.
 - `bint use_cuda_graph`: whether to use CUDA Graphs to improve performance and reduce CPU overhead in CUDA backend. Not all models are supported.
 - `int ml_program`: select CoreML provider.
   - 0: NeuralNetwork
   - 1: MLProgram

When `overlap` and `tilesize` are not specified, the filter will internally try to resize the network to fit the input clips. This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. left out `overlap`, `tilesize` at all and just process the input frame in one tile, or
2. set all three so that the frame is processed in `tilesize[0]` x `tilesize[1]` tiles, and adjacent tiles will have an overlap of `overlap[0]` x `overlap[1]` pixels on each direction. The overlapped region will be throw out so that only internal output pixels are used.
