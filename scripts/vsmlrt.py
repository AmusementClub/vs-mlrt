__version__ = "3.15.5"

__all__ = [
    "Backend", "BackendV2"
    "Waifu2x", "Waifu2xModel",
    "DPIR", "DPIRModel",
    "RealESRGAN", "RealESRGANModel",
    "RealESRGANv2", "RealESRGANv2Model",
    "CUGAN",
    "RIFE", "RIFEModel", "RIFEMerge",
    "inference"
]

import copy
from dataclasses import dataclass, field
import enum
import math
import os
import subprocess
import sys
import tempfile
import time
import typing
import zlib

import vapoursynth as vs
from vapoursynth import core


def get_plugins_path() -> str:
    path = b""

    try:
        path = core.trt.Version()["path"]
    except AttributeError:
        try:
            path = core.ort.Version()["path"]
        except AttributeError:
            try:
                path = core.ov.Version()["path"]
            except AttributeError:
                path = core.ncnn.Version()["path"]

    assert path != b""

    return os.path.dirname(path).decode()

plugins_path: str = get_plugins_path()
trtexec_path: str = os.path.join(plugins_path, "vsmlrt-cuda", "trtexec")
models_path: str = os.path.join(plugins_path, "models")


class Backend:
    @dataclass(frozen=False)
    class ORT_CPU:
        """ backend for cpus """

        num_streams: int = 1
        verbosity: int = 2
        fp16: bool = False
        fp16_blacklist_ops: typing.Optional[typing.Sequence[str]] = None

        # internal backend attributes
        supports_onnx_serialization: bool = True

    @dataclass(frozen=False)
    class ORT_CUDA:
        """ backend for nvidia gpus

        basic performance tuning:
        set fp16 = True (on RTX GPUs)
        """

        device_id: int = 0
        cudnn_benchmark: bool = True
        num_streams: int = 1
        verbosity: int = 2
        fp16: bool = False
        use_cuda_graph: bool = False # preview, not supported by all models
        fp16_blacklist_ops: typing.Optional[typing.Sequence[str]] = None

        # internal backend attributes
        supports_onnx_serialization: bool = True

    @dataclass(frozen=False)
    class OV_CPU:
        """ backend for x86 cpus

        basic performance tuning:
        set bf16 = True (on Zen4)
        increase num_streams
        """

        fp16: bool = False
        num_streams: typing.Union[int, str] = 1
        bind_thread: bool = True
        fp16_blacklist_ops: typing.Optional[typing.Sequence[str]] = None
        bf16: bool = False

        # internal backend attributes
        supports_onnx_serialization: bool = True

    @dataclass(frozen=False)
    class TRT:
        """ backend for nvidia gpus

        basic performance tuning:
        set fp16 = True (on RTX GPUs)
        increase num_streams
        increase workspace
        set use_cuda_graph = True
        """

        max_shapes: typing.Optional[typing.Tuple[int, int]] = None
        opt_shapes: typing.Optional[typing.Tuple[int, int]] = None
        fp16: bool = False
        device_id: int = 0
        workspace: typing.Optional[int] = 128
        verbose: bool = False
        use_cuda_graph: bool = False
        num_streams: int = 1
        use_cublas: bool = False # cuBLAS + cuBLASLt
        static_shape: bool = True
        tf32: bool = True
        log: bool = True

        # as of TensorRT 8.4, it can be turned off without performance penalty in most cases
        use_cudnn: bool = True
        use_edge_mask_convolutions: bool = True
        use_jit_convolutions: bool = True

        heuristic: bool = False # only supported on Ampere+ with TensorRT 8.5+

        output_format: int = 0 # 0: fp32, 1: fp16

        min_shapes: typing.Tuple[int, int] = (0, 0)

        faster_dynamic_shapes: bool = True

        # internal backend attributes
        supports_onnx_serialization: bool = False

    @dataclass(frozen=False)
    class OV_GPU:
        """ backend for nvidia gpus

        basic performance tuning:
        set fp16 = True
        increase num_streams
        """

        fp16: bool = False
        num_streams: typing.Union[int, str] = 1
        device_id: int = 0
        fp16_blacklist_ops: typing.Optional[typing.Sequence[str]] = None

        # internal backend attributes
        supports_onnx_serialization: bool = True

    @dataclass(frozen=False)
    class NCNN_VK:
        """ backend for vulkan devices

        basic performance tuning:
        set fp16 = True (on modern GPUs)
        increase num_streams
        """

        fp16: bool = False
        device_id: int = 0
        num_streams: int = 1

        # internal backend attributes
        supports_onnx_serialization: bool = True


backendT = typing.Union[
    Backend.OV_CPU,
    Backend.ORT_CPU,
    Backend.ORT_CUDA,
    Backend.TRT,
    Backend.OV_GPU,
    Backend.NCNN_VK
]


fallback_backend: typing.Optional[backendT] = None


@enum.unique
class Waifu2xModel(enum.IntEnum):
    anime_style_art = 0
    anime_style_art_rgb = 1
    photo = 2
    upconv_7_anime_style_art_rgb = 3
    upconv_7_photo = 4
    upresnet10 = 5
    cunet = 6


def Waifu2x(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[1, 2] = 2,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1, 2, 3, 4, 5, 6] = 6,
    backend: backendT = Backend.OV_CPU(),
    preprocess: bool = True
) -> vs.VideoNode:

    func_name = "vsmlrt.Waifu2x"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{func_name}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (1, 2):
        raise ValueError(f'{func_name}: "scale" must be 1 or 2')

    if not isinstance(model, int) or model not in Waifu2xModel.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0, 1, 2, 3, 4, 5, or 6')

    if model == 0 and noise == 0:
        raise ValueError(
            f'{func_name}: "anime_style_art" model'
            ' does not support noise reduction level 0'
        )

    if model == 0:
        if clip.format.color_family != vs.GRAY:
            raise ValueError(f'{func_name}: "clip" must be of GRAY color family')
    elif clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if overlap is None:
        overlap_w = overlap_h = [8, 8, 8, 8, 8, 4, 4][model]
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    if model == 6:
        multiple = 4
    else:
        multiple = 1

    width, height = clip.width, clip.height
    if preprocess and model in (0, 1, 2):
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        clip = core.resize.Bicubic(
            clip,
            width * 2, height * 2,
            filter_param_a=0, filter_param_b=0.75
        )

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    folder_path = os.path.join(
        models_path,
        "waifu2x",
        tuple(Waifu2xModel.__members__)[model]
    )

    if model in (0, 1, 2):
        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_model.onnx"
    elif model in (3, 4, 5):
        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_scale2.0x_model.onnx"
    else:
        if scale == 1:
            scale_name = ""
        else:
            scale_name = "scale2.0x_"

        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_{scale_name}model.onnx"

    network_path = os.path.join(folder_path, model_name)

    clip = inference_with_fallback(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    if scale == 1 and clip.width // width == 2:
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        # cr: @AkarinVS
        clip = core.fmtc.resample(
            clip, scale=0.5,
            kernel="impulse", impulse=[-0.1875, 1.375, -0.1875],
            kovrspl=2
        )

    return clip


@enum.unique
class DPIRModel(enum.IntEnum):
    drunet_gray = 0
    drunet_color = 1
    drunet_deblocking_grayscale = 2
    drunet_deblocking_color = 3


def DPIR(
    clip: vs.VideoNode,
    strength: typing.Optional[typing.Union[typing.SupportsFloat, vs.VideoNode]],
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1, 2, 3] = 0,
    backend: backendT = Backend.OV_CPU()
) -> vs.VideoNode:

    func_name = "vsmlrt.DPIR"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if not isinstance(model, int) or model not in DPIRModel.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0, 1, 2 or 3')

    if model in [0, 2] and clip.format.color_family != vs.GRAY:
        raise ValueError(f'{func_name}: "clip" must be of GRAY color family')
    elif model in [1, 3] and clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if strength is None:
        strength = 5.0

    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH

    if isinstance(strength, vs.VideoNode):
        strength = typing.cast(vs.VideoNode, strength)
        if strength.format.color_family != vs.GRAY:
            raise ValueError(f'{func_name}: "strength" must be of GRAY color family')
        if strength.width != clip.width or strength.height != clip.height:
            raise ValueError(f'{func_name}: "strength" must be of the same size as "clip"')
        if strength.num_frames != clip.num_frames:
            raise ValueError(f'{func_name}: "strength" must be of the same length as "clip"')

        strength = core.std.Expr(strength, "x 255 /", format=gray_format)
    else:
        try:
            strength = float(strength)
        except TypeError as e:
            raise TypeError(f'{func_name}: "strength" must be a float or a clip') from e

        strength = core.std.BlankClip(clip, format=gray_format, color=strength / 255, keep=True)

    if overlap is None:
        overlap_w = overlap_h = 0
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 8

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    network_path = os.path.join(
        models_path,
        "dpir",
        f"{tuple(DPIRModel.__members__)[model]}.onnx"
    )

    clip = inference_with_fallback(
        clips=[clip, strength], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    return clip


@enum.unique
class RealESRGANModel(enum.IntEnum):
    # v2
    animevideo_xsx2 = 0
    animevideo_xsx4 = 1
    # v3
    animevideov3 = 2 # 4x

RealESRGANv2Model = RealESRGANModel


def RealESRGAN(
    clip: vs.VideoNode,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1, 2] = 0,
    backend: backendT = Backend.OV_CPU(),
    scale: typing.Optional[float] = None
) -> vs.VideoNode:

    func_name = "vsmlrt.RealESRGAN"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if not isinstance(model, int) or model not in RealESRGANv2Model.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0, 1 or 2')

    if overlap is None:
        overlap_w = overlap_h = 8
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 1

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    if model in [0, 1]:
        network_path = os.path.join(
            models_path,
            "RealESRGANv2",
            f"RealESRGANv2-{tuple(RealESRGANv2Model.__members__)[model]}.onnx".replace('_', '-')
        )
    elif model == 2:
        network_path = os.path.join(
            models_path,
            "RealESRGANv2",
            "realesr-animevideov3.onnx"
        )

    clip_org = clip
    clip = inference_with_fallback(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    if scale is not None:
        scale_h = clip.width // clip_org.width
        scale_v = clip.height // clip_org.height

        assert scale_h == scale_v

        if scale != scale_h:
            rescale = scale / scale_h

            if rescale > 1:
                clip = core.fmtc.resample(clip, scale=rescale, kernel="lanczos", taps=4)
            else:
                clip = core.fmtc.resample(clip, scale=rescale, kernel="lanczos", taps=4, fh=1/rescale, fv=1/rescale)

    return clip

RealESRGANv2 = RealESRGAN


def CUGAN(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[2, 3, 4] = 2,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    backend: backendT = Backend.OV_CPU(),
    preprocess: bool = True,
    alpha: float = 1.0,
    version: typing.Literal[1, 2] = 1, # 1: legacy, 2: pro
    conformance: bool = True # currently specifies dynamic range compression for cugan-pro
) -> vs.VideoNode:
    """
    denoising strength: 0 < -1 < 1 < 2 < 3

    version: (1 or 2)
        1 -> legacy,
        2 -> pro (only models for "noise" in [-1, 0, 3] and "scale" in [2, 3] are published currently)
    """

    func_name = "vsmlrt.CUGAN"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{func_name}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (2, 3, 4):
        raise ValueError(f'{func_name}: "scale" must be 2, 3 or 4')

    if scale != 2 and noise in [1, 2]:
        raise ValueError(
            f'{func_name}: "scale={scale}" model'
            f' does not support noise reduction level {noise}'
        )

    if clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if overlap is None:
        overlap_w = overlap_h = 4
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 2

    width, height = clip.width, clip.height

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    folder_path = os.path.join(models_path, "cugan")

    if version == 1:
        if noise == -1:
            model_name = f"up{scale}x-latest-no-denoise.onnx"
        elif noise == 0:
            model_name = f"up{scale}x-latest-conservative.onnx"
        else:
            model_name = f"up{scale}x-latest-denoise{noise}x.onnx"
    elif version == 2:
        if noise == -1:
            model_name = f"pro-no-denoise3x-up{scale}x.onnx"
        elif noise == 0:
            model_name = f"pro-conservative-up{scale}x.onnx"
        else:
            model_name = f"pro-denoise{noise}x-up{scale}x.onnx"
    else:
        raise ValueError(f'{func_name}: unknown version ({version}), must be 1 (legacy) or 2 (pro)')

    network_path = os.path.join(folder_path, model_name)

    # https://github.com/bilibili/ailab/blob/978f3be762183d7fa79525f29a43e65afb995f6b/Real-CUGAN/upcunet_v3.py#L207
    # mutates network_path
    if alpha != 1.0:
        alpha = float(alpha)

        import numpy as np
        import onnx
        from onnx import numpy_helper

        model = onnx.load(network_path)

        for idx, node in reversed(list(enumerate(model.graph.node))):
            if node.op_type == "ConvTranspose":
                break

        upstream_name = node.input[0]
        downstream_name = node.input[0] + "_mul"
        node.input[0] = downstream_name

        alpha_array = np.array(alpha, dtype=np.float32)
        alpha_tensor = numpy_helper.from_array(alpha_array)
        alpha_constant = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["alpha"],
            value=alpha_tensor
        )
        model.graph.node.insert(idx, alpha_constant)

        mul_node = onnx.helper.make_node(
            "Mul",
            inputs=[upstream_name, "alpha"],
            outputs=[downstream_name]
        )
        model.graph.node.insert(idx+1, mul_node)

        if backend.supports_onnx_serialization:
            if conformance and version == 2:
                clip = core.std.Expr(clip, "x 0.7 * 0.15 +")

            clip = inference_with_fallback(
                clips=[clip], network_path=model.SerializeToString(),
                overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
                backend=backend, path_is_serialization=True
            )

            if conformance and version == 2:
                clip = core.std.Expr(clip, "x 0.15 - 0.7 /")

            return clip

        network_path = f"{network_path}_alpha{alpha!r}.onnx"
        onnx.save(model, network_path)

    # https://github.com/bilibili/ailab/blob/e102bef22384c629f82552dbec3d6b5bab125639/Real-CUGAN/upcunet_v3.py#L1275-L1276
    if conformance and version == 2:
        clip = core.std.Expr(clip, "x 0.7 * 0.15 +")

    clip = inference_with_fallback(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    # https://github.com/bilibili/ailab/blob/e102bef22384c629f82552dbec3d6b5bab125639/Real-CUGAN/upcunet_v3.py#L269
    if conformance and version == 2:
        clip = core.std.Expr(clip, "x 0.15 - 0.7 /")

    return clip


def get_rife_input(clip: vs.VideoNode) -> typing.List[vs.VideoNode]:
    empty = clip.std.BlankClip(format=vs.GRAYS, length=1)

    if hasattr(core, 'akarin'):
        horizontal = as_bits(core.akarin.Expr(empty, 'X 2 * width 1 - / 1 -'), clip)
        vertical = as_bits(core.akarin.Expr(empty, 'Y 2 * height 1 - / 1 -'), clip)
    else:
        from functools import partial

        def meshgrid_core(n: int, f: vs.VideoFrame, horizontal: bool) -> vs.VideoFrame:
            fout = f.copy()

            is_api4 = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4
            if is_api4:
                mem_view = fout[0]
            else:
                mem_view = fout.get_write_array(0)

            height, width = mem_view.shape

            if horizontal:
                for i in range(height):
                    for j in range(width):
                        mem_view[i, j] = 2 * j / (width - 1) - 1
            else:
                for i in range(height):
                    for j in range(width):
                        mem_view[i, j] = 2 * i / (height - 1) - 1

            return fout

        horizontal = as_bits(core.std.ModifyFrame(empty, empty, partial(meshgrid_core, horizontal=True)), clip)
        vertical = as_bits(core.std.ModifyFrame(empty, empty, partial(meshgrid_core, horizontal=False)), clip)

    horizontal = horizontal.std.Loop(clip.num_frames)
    vertical = vertical.std.Loop(clip.num_frames)

    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH

    multiplier_h = clip.std.BlankClip(format=gray_format, color=2/(clip.width-1), keep=True)

    multiplier_w = clip.std.BlankClip(format=gray_format, color=2/(clip.height-1), keep=True)

    return [horizontal, vertical, multiplier_h, multiplier_w]


@enum.unique
class RIFEModel(enum.IntEnum):
    v4_0 = 40
    v4_2 = 42
    v4_3 = 43
    v4_4 = 44
    v4_5 = 45
    v4_6 = 46


def RIFEMerge(
    clipa: vs.VideoNode,
    clipb: vs.VideoNode,
    mask: vs.VideoNode,
    scale: float = 1.0,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[40, 42, 43, 44, 45, 46] = 44,
    backend: backendT = Backend.OV_CPU()
) -> vs.VideoNode:
    """ temporal MaskedMerge-like interface for the RIFE model

    Its semantics is similar to core.std.MaskedMerge(clipa, clipb, mask, first_plane=True),
    except that it merges the two clips in the time domain and you specify the "mask" based
    on the time point of the resulting clip (range (0,1)) between the two clips.
    """

    from fractions import Fraction

    func_name = "vsmlrt.RIFEMerge"

    for clip in (clipa, clipb, mask):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f'{func_name}: clip must be a clip!')

        if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
            raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    for clip in (clipa, clipb):
        if clip.format.color_family != vs.RGB:
            raise ValueError(f'{func_name}: "clipa" / "clipb" must be of RGB color family')

        if clip.width != mask.width or clip.height != mask.height:
            raise ValueError(f'{func_name}: video dimensions mismatch')

        if clip.num_frames != mask.num_frames:
            raise ValueError(f'{func_name}: number of frames mismatch')

    if mask.format.color_family != vs.GRAY:
        raise ValueError(f'{func_name}: "mask" must be of GRAY color family')

    if tiles is not None or tilesize is not None or overlap is not None:
        raise ValueError(f'{func_name}: tiling is not supported')

    if overlap is None:
        overlap_w = overlap_h = 0
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple_frac = 32 / Fraction(scale)
    if multiple_frac.denominator != 1:
        raise ValueError(f'{func_name}: (32 / Fraction(scale)) must be an integer')
    multiple = int(multiple_frac.numerator)
    scale = float(Fraction(scale))

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    network_path = os.path.join(
        models_path,
        "rife",
        f"rife_v{model // 10}.{model % 10}.onnx"
    )

    clips = [clipa, clipb, mask, *get_rife_input(clipa)]

    if scale == 1.0:
        return inference_with_fallback(
            clips=clips, network_path=network_path,
            overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
            backend=backend
        )
    else:
        import onnx
        from onnx.numpy_helper import from_array, to_array

        onnx_model = onnx.load(network_path)

        resize_counter = 0
        for i in range(len(onnx_model.graph.node)):
            node = onnx_model.graph.node[i]
            if len(node.output) == 1 and node.op_type == "Constant" and node.output[0].startswith("onnx::Resize"):
                resize_counter += 1

                array = to_array(node.attribute[0].t).copy()
                if resize_counter % 3 == 2:
                    array[2:4] /= scale
                else:
                    array[2:4] *= scale
                onnx_model.graph.node[i].attribute[0].t.raw_data = from_array(array).raw_data

        if resize_counter != 11:
            raise ValueError("invalid rife model")

        multiplier_counter = 0
        for i in range(len(onnx_model.graph.node)):
            node = onnx_model.graph.node[i]
            if len(node.output) == 1 and node.op_type == "Constant" and node.output[0].startswith("onnx::Mul"):
                multiplier_counter += 1

                array = to_array(node.attribute[0].t).copy()
                if multiplier_counter % 2 == 1:
                    array /= scale
                else:
                    array *= scale
                onnx_model.graph.node[i].attribute[0].t.raw_data = from_array(array).raw_data

        if multiplier_counter != 7:
            raise ValueError("invalid rife model")

        if backend.supports_onnx_serialization:
            return inference_with_fallback(
                clips=clips, network_path=onnx_model.SerializeToString(),
                overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
                backend=backend, path_is_serialization=True
            )
        else:
            network_path = f"{network_path}_scale{scale!r}.onnx"
            onnx.save(onnx_model, network_path)

            return inference_with_fallback(
                clips=clips, network_path=network_path,
                overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
                backend=backend
            )


def RIFE(
    clip: vs.VideoNode,
    multi: int = 2,
    scale: float = 1.0,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[40, 42, 43, 44, 45, 46] = 44,
    backend: backendT = Backend.OV_CPU()
) -> vs.VideoNode:
    """ RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    multi, scale is based on vs-rife.

    For the best results, you need to perform scene detection on the input clip
    (e.g. misc.SCDetect, mv.SCDetection) before passing it to RIFE.
    Also note that the quality of result is strongly dependent on high quality
    scene detection and you might need to tweak the scene detection parameters
    and/or filter to achieve the best quality.

    Args:
        multi: Multiple of the frame counts.
            Default: 2.

        scale: Controls the process resolution for optical flow model.
            32 / fractions.Fraction(scale) must be an integer.
            scale=0.5 is recommended for 4K video.
    """

    func_name = "vsmlrt.RIFE"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if multi < 2:
        raise ValueError(f'{func_name}: RIFE: multi must be at least 2')

    if tiles is not None or tilesize is not None or overlap is not None:
        raise ValueError(f'{func_name}: tiling is not supported')

    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH

    initial = core.std.Interleave([clip] * (multi - 1))

    terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    terminal = core.std.Interleave([terminal] * (multi - 1))

    timepoint = core.std.Interleave([
        clip.std.BlankClip(format=gray_format, color=i/multi, length=1)
        for i in range(1, multi)
    ]).std.Loop(clip.num_frames)

    output0 = RIFEMerge(
        clipa=initial, clipb=terminal, mask=timepoint,
        scale=scale, tiles=tiles, tilesize=tilesize, overlap=overlap,
        model=model, backend=backend
    )

    clip = as_bits(clip, output0)
    initial = core.std.Interleave([clip] * (multi - 1))

    if hasattr(core, 'akarin') and hasattr(core.akarin, 'Select'):
        output = core.akarin.Select([output0, initial], initial, 'x._SceneChangeNext 1 0 ?')
    else:
        def handler(n: int, f: vs.VideoFrame) -> vs.VideoNode:
            if f.props.get('_SceneChangeNext'):
                return initial
            return output0
        output = core.std.FrameEval(output0, handler, initial)

    if multi == 2:
        return core.std.Interleave([clip, output])
    else:
        return core.std.Interleave([
            clip,
            *(output.std.SelectEvery(cycle=multi-1, offsets=i) for i in range(multi - 1))
        ])


def get_engine_path(
    network_path: str,
    min_shapes: typing.Tuple[int, int],
    opt_shapes: typing.Tuple[int, int],
    max_shapes: typing.Tuple[int, int],
    workspace: typing.Optional[int],
    fp16: bool,
    device_id: int,
    use_cublas: bool,
    static_shape: bool,
    tf32: bool,
    use_cudnn: bool,
    input_format: int,
    output_format: int
) -> str:

    with open(network_path, "rb") as file:
        checksum = zlib.adler32(file.read())

    trt_version = core.trt.Version()["tensorrt_version"].decode()

    try:
        device_name = core.trt.DeviceProperties(device_id)["name"].decode()
        device_name = device_name.replace(' ', '-')
    except AttributeError:
        device_name = f"device{device_id}"

    if static_shape:
        shape_str = f".{opt_shapes[0]}x{opt_shapes[1]}"
    else:
        shape_str = (
            f".min{min_shapes[0]}x{min_shapes[1]}"
            f"_opt{opt_shapes[0]}x{opt_shapes[1]}"
            f"_max{max_shapes[0]}x{max_shapes[1]}"
        )

    return (
        network_path +
        shape_str +
        ("_fp16" if fp16 else "") +
        ("_no-tf32" if not tf32 else "") +
        (f"_workspace{workspace}" if workspace is not None else "") +
        f"_trt-{trt_version}" +
        ("_cublas" if use_cublas else "") +
        ("_cudnn" if use_cudnn else "") +
        "_I-" + ("fp32" if input_format == 0 else "fp16") +
        "_O-" + ("fp32" if output_format == 0 else "fp16") +
        f"_{device_name}" +
        f"_{checksum:x}" +
        ".engine"
    )


def trtexec(
    network_path: str,
    channels: int,
    opt_shapes: typing.Tuple[int, int],
    max_shapes: typing.Tuple[int, int],
    fp16: bool,
    device_id: int,
    workspace: typing.Optional[int] = 128,
    verbose: bool = False,
    use_cuda_graph: bool = False,
    use_cublas: bool = False,
    static_shape: bool = True,
    tf32: bool = True,
    log: bool = False,
    use_cudnn: bool = True,
    use_edge_mask_convolutions: bool = True,
    use_jit_convolutions: bool = True,
    heuristic: bool = False,
    input_name: str = "input",
    input_format: int = 0,
    output_format: int = 0,
    min_shapes: typing.Tuple[int, int] = (0, 0),
    faster_dynamic_shapes: bool = True
) -> str:

    # tensort runtime version, e.g. 8401 => 8.4.1
    trt_version = int(core.trt.Version()["tensorrt_version"])

    if isinstance(opt_shapes, int):
        opt_shapes = (opt_shapes, opt_shapes)

    if isinstance(max_shapes, int):
        max_shapes = (max_shapes, max_shapes)

    engine_path = get_engine_path(
        network_path=network_path,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        workspace=workspace,
        fp16=fp16,
        device_id=device_id,
        use_cublas=use_cublas,
        static_shape=static_shape,
        tf32=tf32,
        use_cudnn=use_cudnn,
        input_format=input_format,
        output_format=output_format
    )

    if os.access(engine_path, mode=os.R_OK):
        return engine_path

    alter_engine_path = os.path.join(
        tempfile.gettempdir(),
        os.path.splitdrive(engine_path)[1][1:]
    )

    if os.access(alter_engine_path, mode=os.R_OK):
        return alter_engine_path

    try:
        # test writability
        with open(engine_path, "w") as f:
            pass
        os.remove(engine_path)
    except PermissionError:
        print(f"{engine_path} not writable", file=sys.stderr)
        engine_path = alter_engine_path
        dirname = os.path.dirname(engine_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print(f"change engine path to {engine_path}", file=sys.stderr)

    args = [
        trtexec_path,
        f"--onnx={network_path}",
        f"--timingCacheFile={engine_path}.cache",
        f"--device={device_id}",
        f"--saveEngine={engine_path}"
    ]

    if workspace is not None:
        if trt_version >= 8400:
            args.append(f"--memPoolSize=workspace:{workspace}")
        else:
            args.append(f"--workspace{workspace}")

    if static_shape:
        args.append(f"--shapes={input_name}:1x{channels}x{opt_shapes[1]}x{opt_shapes[0]}")
    else:
        args.extend([
            f"--minShapes=input:1x{channels}x{min_shapes[1]}x{min_shapes[0]}",
            f"--optShapes=input:1x{channels}x{opt_shapes[1]}x{opt_shapes[0]}",
            f"--maxShapes=input:1x{channels}x{max_shapes[1]}x{max_shapes[0]}"
        ])

    if fp16:
        args.append("--fp16")

    if verbose:
        args.append("--verbose")

    disabled_tactic_sources = []
    if not use_cublas:
        disabled_tactic_sources.extend(["-CUBLAS", "-CUBLAS_LT"])
    if not use_cudnn:
        disabled_tactic_sources.append("-CUDNN")
    if not use_edge_mask_convolutions and trt_version >= 8401:
        disabled_tactic_sources.append("-EDGE_MASK_CONVOLUTIONS")
    if not use_jit_convolutions and trt_version >= 8500:
        disabled_tactic_sources.append("-JIT_CONVOLUTIONS")
    if disabled_tactic_sources:
        args.append(f"--tacticSources={','.join(disabled_tactic_sources)}")

    if use_cuda_graph:
        args.extend((
            "--useCudaGraph",
            "--noDataTransfers"
        ))
    else:
        args.append("--buildOnly")

    if not tf32:
        args.append("--noTF32")

    if heuristic and trt_version >= 8500 and core.trt.DeviceProperties(device_id)["major"] >= 8:
        args.append("--heuristic")

    args.extend([
        "--inputIOFormats=fp32:chw" if input_format == 0 else "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp32:chw" if output_format == 0 else "--outputIOFormats=fp16:chw"
    ])

    if faster_dynamic_shapes and not static_shape and trt_version >= 8500:
        args.append("--preview=+fasterDynamicShapes0805")

    if log:
        env_key = "TRTEXEC_LOG_FILE"
        prev_env_value = os.environ.get(env_key)

        if prev_env_value is not None and len(prev_env_value) > 0:
            # env_key has been set, no extra action
            env = {env_key: prev_env_value}
            subprocess.run(args, env=env, check=True, stdout=sys.stderr)
        else:
            time_str = time.strftime('%y%m%d_%H%M%S', time.localtime())

            log_filename = os.path.join(
                tempfile.gettempdir(),
                f"trtexec_{time_str}.log"
            )

            env = {env_key: log_filename}

            completed_process = subprocess.run(args, env=env, check=False, stdout=sys.stderr)

            if completed_process.returncode == 0:
                try:
                    os.remove(log_filename)
                except FileNotFoundError:
                    # maybe the official trtexec is used?
                    pass
            else:
                if os.path.exists(log_filename):
                    raise RuntimeError(f"trtexec execution fails, log has been written to {log_filename}")
                else:
                    raise RuntimeError(f"trtexec execution fails but no log is found")
    else:
        subprocess.run(args, check=True, stdout=sys.stderr)

    return engine_path


def calc_size(width: int, tiles: int, overlap: int, multiple: int = 1) -> int:
    return math.ceil((width + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple


def calc_tilesize(
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    width: int,
    height: int,
    multiple: int,
    overlap_w: int,
    overlap_h: int
) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:

    if tilesize is None:
        if tiles is None:
            overlap_w = 0
            overlap_h = 0
            tile_w = width
            tile_h = height
        elif isinstance(tiles, int):
            tile_w = calc_size(width, tiles, overlap_w, multiple)
            tile_h = calc_size(height, tiles, overlap_h, multiple)
        else:
            tile_w = calc_size(width, tiles[0], overlap_w, multiple)
            tile_h = calc_size(height, tiles[1], overlap_h, multiple)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    return (tile_w, tile_h), (overlap_w, overlap_h)


def init_backend(
    backend: backendT,
    trt_opt_shapes: typing.Tuple[int, int]
) -> backendT:

    if backend is Backend.ORT_CPU: # type: ignore
        backend = Backend.ORT_CPU()
    elif backend is Backend.ORT_CUDA: # type: ignore
        backend = Backend.ORT_CUDA()
    elif backend is Backend.OV_CPU: # type: ignore
        backend = Backend.OV_CPU()
    elif backend is Backend.TRT: # type: ignore
        backend = Backend.TRT()
    elif backend is Backend.OV_GPU: # type: ignore
        backend = Backend.OV_GPU()
    elif backend is Backend.NCNN_VK: # type: ignore
        backend = Backend.NCNN_VK()

    backend = copy.deepcopy(backend)

    if isinstance(backend, Backend.TRT):
        if backend.opt_shapes is None:
            backend.opt_shapes = trt_opt_shapes

        if backend.max_shapes is None:
            backend.max_shapes = backend.opt_shapes

    return backend


def _inference(
    clips: typing.List[vs.VideoNode],
    network_path: typing.Union[bytes, str],
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: backendT,
    path_is_serialization: bool = False,
    input_name: str = "input"
) -> vs.VideoNode:

    if not path_is_serialization:
        network_path = typing.cast(str, network_path)
        if not os.path.exists(network_path):
            raise RuntimeError(
                f'"{network_path}" not found, '
                f'built-in models can be found at https://github.com/AmusementClub/vs-mlrt/releases'
            )

    if isinstance(backend, Backend.ORT_CPU):
        clip = core.ort.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            provider="CPU", builtin=False,
            num_streams=backend.num_streams,
            verbosity=backend.verbosity,
            fp16=backend.fp16,
            path_is_serialization=path_is_serialization,
            fp16_blacklist_ops=backend.fp16_blacklist_ops
        )
    elif isinstance(backend, Backend.ORT_CUDA):
        clip = core.ort.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            provider="CUDA", builtin=False,
            device_id=backend.device_id,
            num_streams=backend.num_streams,
            verbosity=backend.verbosity,
            cudnn_benchmark=backend.cudnn_benchmark,
            fp16=backend.fp16,
            path_is_serialization=path_is_serialization,
            use_cuda_graph=backend.use_cuda_graph,
            fp16_blacklist_ops=backend.fp16_blacklist_ops
        )
    elif isinstance(backend, Backend.OV_CPU):
        config = lambda: dict(
            CPU_THROUGHPUT_STREAMS=backend.num_streams,
            CPU_BIND_THREAD="YES" if backend.bind_thread else "NO",
            ENFORCE_BF16="YES" if backend.bf16 else "NO"
        )

        clip = core.ov.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            device="CPU", builtin=False,
            fp16=backend.fp16,
            config=config,
            path_is_serialization=path_is_serialization,
            fp16_blacklist_ops=backend.fp16_blacklist_ops
        )
    elif isinstance(backend, Backend.OV_GPU):
        config = lambda: dict(
            GPU_THROUGHPUT_STREAMS=backend.num_streams
        )
        clip = core.ov.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            device=f"GPU.{backend.device_id}", builtin=False,
            fp16=backend.fp16,
            config=config,
            path_is_serialization=path_is_serialization,
            fp16_blacklist_ops=backend.fp16_blacklist_ops
        )
    elif isinstance(backend, Backend.TRT):
        if path_is_serialization:
            raise ValueError('"path_is_serialization" must be False for trt backend')

        network_path = typing.cast(str, network_path)

        channels = sum(clip.format.num_planes for clip in clips)

        opt_shapes = backend.opt_shapes if backend.opt_shapes is not None else tilesize
        max_shapes = backend.max_shapes if backend.max_shapes is not None else tilesize

        engine_path = trtexec(
            network_path,
            channels=channels,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
            fp16=backend.fp16,
            device_id=backend.device_id,
            workspace=backend.workspace,
            verbose=backend.verbose,
            use_cuda_graph=backend.use_cuda_graph,
            use_cublas=backend.use_cublas,
            static_shape=backend.static_shape,
            tf32=backend.tf32,
            log=backend.log,
            use_cudnn=backend.use_cudnn,
            use_edge_mask_convolutions=backend.use_edge_mask_convolutions,
            use_jit_convolutions=backend.use_jit_convolutions,
            heuristic=backend.heuristic,
            input_name=input_name,
            input_format=clips[0].format.bits_per_sample == 16,
            output_format=backend.output_format,
            min_shapes=backend.min_shapes,
            faster_dynamic_shapes=backend.faster_dynamic_shapes
        )
        clip = core.trt.Model(
            clips, engine_path,
            overlap=overlap,
            tilesize=tilesize,
            device_id=backend.device_id,
            use_cuda_graph=backend.use_cuda_graph,
            num_streams=backend.num_streams,
            verbosity=4 if backend.verbose else 2
        )
    elif isinstance(backend, Backend.NCNN_VK):
        clip = core.ncnn.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            device_id=backend.device_id,
            num_streams=backend.num_streams,
            builtin=False,
            fp16=backend.fp16,
            path_is_serialization=path_is_serialization,
        )
    else:
        raise TypeError(f'unknown backend {backend}')

    return clip


def inference_with_fallback(
    clips: typing.List[vs.VideoNode],
    network_path: typing.Union[bytes, str],
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: backendT,
    path_is_serialization: bool = False,
    input_name: str = "input"
) -> vs.VideoNode:

    try:
        return _inference(
            clips=clips, network_path=network_path,
            overlap=overlap, tilesize=tilesize,
            backend=backend,
            path_is_serialization=path_is_serialization,
            input_name=input_name
        )
    except Exception as e:
        if fallback_backend is not None:
            import logging
            logger = logging.getLogger("vsmlrt")
            logger.warning(f'"{backend}" fails, trying fallback backend "{fallback_backend}"')

            return _inference(
                clips=clips, network_path=network_path,
                overlap=overlap, tilesize=tilesize,
                backend=fallback_backend,
                path_is_serialization=path_is_serialization,
                input_name=input_name
            )
        else:
            raise e


def inference(
    clips: typing.Union[vs.VideoNode, typing.List[vs.VideoNode]],
    network_path: str,
    overlap: typing.Tuple[int, int] = (0, 0),
    tilesize: typing.Optional[typing.Tuple[int, int]] = None,
    backend: backendT = Backend.OV_CPU(),
    input_name: typing.Optional[str] = "input"
) -> vs.VideoNode:

    if isinstance(clips, vs.VideoNode):
        clips = typing.cast(vs.VideoNode, clips)
        clips = [clips]

    if tilesize is None:
        tilesize = (clips[0].width, clips[0].height)

    backend = init_backend(backend=backend, trt_opt_shapes=tilesize)

    if input_name is None:
        input_name = get_input_name(network_path)

    return inference_with_fallback(
        clips=clips,
        network_path=network_path,
        overlap=overlap,
        tilesize=tilesize,
        backend=backend,
        path_is_serialization=False,
        input_name=input_name
    )


def get_input_name(network_path: str) -> str:
    import onnx
    model = onnx.load(network_path)
    return model.graph.input[0].name


def as_bits(clip: vs.VideoNode, target: vs.VideoNode) -> vs.VideoNode:
    if clip.format.bits_per_sample == target.format.bits_per_sample:
        return clip
    else:
        is_api4 = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4
        query_video_format = core.query_video_format if is_api4 else core.register_format
        format = query_video_format(
            color_family=clip.format.color_family,
            sample_type=clip.format.sample_type,
            bits_per_sample=target.format.bits_per_sample,
            subsampling_w=clip.format.subsampling_w,
            subsampling_h=clip.format.subsampling_h
        )
        return clip.resize.Point(format=format)


class BackendV2:
    """ simplified backend interfaces with keyword-only arguments

    More exposed arguments may be added for each backend,
    but existing ones will always function in a forward compatible way.
    """

    @staticmethod
    def TRT(*,
        num_streams: int = 1,
        fp16: bool = False,
        tf32: bool = True,
        output_format: int = 0, # 0: fp32, 1: fp16
        workspace: typing.Optional[int] = 128,
        use_cuda_graph: bool = False,
        static_shape: bool = True,
        min_shapes: typing.Tuple[int, int] = (0, 0),
        opt_shapes: typing.Optional[typing.Tuple[int, int]] = None,
        max_shapes: typing.Optional[typing.Tuple[int, int]] = None,
        use_cublas: bool = False,
        use_cudnn: bool = True,
        device_id: int = 0,
        **kwargs
    ) -> Backend.TRT:

        return Backend.TRT(
            num_streams=num_streams,
            fp16=fp16, tf32=tf32, output_format=output_format,
            workspace=workspace, use_cuda_graph=use_cuda_graph,
            static_shape=static_shape,
            min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes,
            use_cublas=use_cublas, use_cudnn=use_cudnn,
            device_id=device_id,
            **kwargs
        )

    @staticmethod
    def NCNN_VK(*,
        num_streams: int = 1,
        fp16: bool = False,
        device_id: int = 0,
        **kwargs
    ) -> Backend.NCNN_VK:
        return Backend.NCNN_VK(
            num_streams=num_streams,
            fp16=fp16,
            device_id=device_id,
            **kwargs
        )

    @staticmethod
    def ORT_CUDA(*,
        num_streams: int = 1,
        fp16: bool = False,
        cudnn_benchmark: bool = True,
        device_id: int = 0,
        **kwargs
    ) -> Backend.ORT_CUDA:
        return Backend.ORT_CUDA(
            num_streams=num_streams,
            fp16=fp16,
            cudnn_benchmark=cudnn_benchmark,
            device_id=device_id,
            **kwargs
        )

    @staticmethod
    def OV_CPU(*,
        num_streams: typing.Union[int, str] = 1,
        bf16: bool = False,
        bind_thread: bool = True,
        **kwargs
    ) -> Backend.OV_CPU:
        return Backend.OV_CPU(
            num_streams=num_streams,
            bf16=bf16,
            bind_thread=bind_thread,
            **kwargs
        )

    @staticmethod
    def ORT_CPU(*,
        num_streams: int = 1,
        **kwargs
    ) -> Backend.ORT_CPU:
        return Backend.ORT_CPU(
            num_streams=num_streams,
            **kwargs
        )

    @staticmethod
    def OV_GPU(*,
        num_streams: typing.Union[int, str] = 1,
        fp16: bool = False,
        device_id: int = 0,
        **kwargs
    ) -> Backend.OV_GPU:
        return Backend.OV_GPU(
            num_streams=num_streams,
            fp16=fp16,
            device_id=device_id,
            **kwargs
        )
