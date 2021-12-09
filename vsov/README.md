# VapourSynth OpenVINO

The vs-openvino plugin provides optimized *pure* CPU runtime for some popular AI filters.

## Building and Installation

To build, you will need [OpenVINO](https://docs.openvino.ai/latest/get_started.html) and its dependencies.
Only `Model Optimizer` and `Inference Engine` are required.

You can download official Intel releases:
- [Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- [Windows](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)
- [macOS](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_macos.html)

Or, you can use our prebuilt Windows binary releases from [AmusementClub](https://github.com/AmusementClub/openvino/releases/latest/), our release has the benefit of static linking support.

Sample cmake commands to build:
```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
	-D CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
	-D InferenceEngine_DIR=openvino/runtime/cmake
	-D VAPOURSYNTH_INCLUDE_DIRECTORY="path/to/vapoursynth/include"
cmake --build build
cmake --install build --prefix install
```
You should find `vsov.dll` (or libvsov.so) under `install/bin`. You will also need Intel TBB (you can get
`tbb.dll` from OpenVINO release). On windows, `tbb.dll` must be placed under `vapoursynth/plugins/vsov/`
directory for `vsov.dll` to find.

## Usage

Prototype: `core.ov.Model(clip[] clips, string network_path[, int pad = 0, int block_w = 0, int block_h = 0, string device = "CPU", bint builtin = 0, string builtindir="models"])`

Arguments:
 - `clip[] clips`: the input clips, only 32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
 - `string network_path`: the path to the network in ONNX format.
 - `int pad`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `pad` argument specifies the overlapping (both horizontal and vertical, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended padding size.
 - `int block_w`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the horizontal tile size (including the padding). Please refer to network specific docs on the recommended tile size.
 - `int block_h`: Similar to `block_w`, this set the height of the tile. If unspecified, it will default to `block_w`.
 - `string device`: Specifies the device to run the inference on. Currently only `"CPU"` is supported, which is also the default.
 - `bint builtin`: whether to load the model from the VS plugins directory, see also `builtindir`.
 - `string builtindir`: the model directory under VS plugins directory for builtin models, default "models".

When `pad = 0` (which is the default), the filter will internally try to resize the network to fit the input clips. This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. left out `pad`, `block_w`, `block_h` at all and just process the input frame in one tile, or
2. set all three so that the frame is processed in `block_w` x `block_h` tiles, and adjacent tiles will have an overlap of `pad` pixels on both directions. The overlapped region will be throw out so that only internal output pixels are used..
