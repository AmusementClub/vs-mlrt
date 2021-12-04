# VapourSynth ONNX Runtime

The vs-onnxruntime plugin provides optimized CPU & CUDA runtime for some popular AI filters.

## Building and Installation

To build, you will need [ONNX Runtime](https://www.onnxruntime.ai/), [protobuf](https://github.com/protocolbuffers/protobuf), [ONNX](https://github.com/onnx/onnx) and thier dependencies.

Please refer to [ONNX Runtime Docs](https://onnxruntime.ai/docs/install/) for installation notes.
Or, you can use our prebuilt Windows binary releases from [AmusementClub](https://github.com/AmusementClub/onnxruntime/releases/latest/).

Please refer to our [github actions workflow](../.github/workflows/windows-ort.yml) for sample building instructions.

## Usage

Prototype: `core.ort.Model(clip[] clips, string network_path[, int pad = 0, int block_w = 0, int block_h = 0, string provider = "", int device_id = 0, int verbosity = 2])`

Arguments:
 - `clip[] clips`: the input clips, only 32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
 - `string network_path`: the path to the network in ONNX format.
 - `int pad`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `pad` argument specifies the overlapping (both horizontal and vertical, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended padding size.
 - `int block_w`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the horizontal tile size (including the padding). Please refer to network specific docs on the recommended tile size.
 - `int block_h`: Similar to `block_w`, this set the height of the tile. If unspecified, it will default to `block_w`.
 - `string provider`: Specifies the device to run the inference on.
   - `"CPU"` or `""`: pure CPU backend
   - `"CUDA"`: CUDA GPU backend
 - `int device_id`: select the GPU device for the CUDA backend.
 - `int verbosity`: specify the verbosity of logging, the default is warning.
   - 0: fatal error only, `ORT_LOGGING_LEVEL_FATAL`
   - 1: also errors, `ORT_LOGGING_LEVEL_ERROR`
   - 2: also warnings, `ORT_LOGGING_LEVEL_WARNING`
   - 3: also info, `ORT_LOGGING_LEVEL_INFO`
   - 4: everything, `ORT_LOGGING_LEVEL_VERBOSE`

When `pad = 0` (which is the default), the filter will internally try to resize the network to fit the input clips. This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. left out `pad`, `block_w`, `block_h` at all and just process the input frame in one tile, or
2. set all three so that the frame is processed in `block_w` x `block_h` tiles, and adjacent tiles will have an overlap of `pad` pixels on both directions. The overlapped region will be throw out so that only internal output pixels are used.
