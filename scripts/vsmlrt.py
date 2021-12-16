import enum
import os.path
import typing

import vapoursynth as vs
from vapoursynth import core


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
    block_w: typing.Optional[int] = None,
    block_h: typing.Optional[int] = None,
    pad: int = 0,
    model: typing.Literal[0, 1, 2, 3, 4, 5, 6] = 6,
    backend: typing.Literal["ort-cpu", "ort-cuda", "ov-cpu"] = "ort-cpu",
    # parameters for "ort-cuda"
    device_id: int = 0,
    cudnn_benchmark: bool = True
) -> vs.VideoNode:

    funcName = "vsmlrt.Waifu2x"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{funcName}: only constant format 32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{funcName}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (1, 2):
        raise ValueError(f'{funcName}: "scale" must be 1 or 2')

    if not isinstance(model, int) or model not in Waifu2xModel.__members__.values():
        raise ValueError(f'{funcName}: "model" must be 0, 1, 2, 3, 4, 5, or 6')

    if model == 0 and noise == 0:
        raise ValueError(
            f'{funcName}: "anime_style_art" model'
            ' does not support noise reduction level 0'
        )

    if block_h is None:
        if block_w is None:
            block_w = clip.width
            block_h = clip.height
        else:
            block_h = block_w

    if model == 6 and (block_w % 4 != 0 or block_h % 4 != 0): # type: ignore
        raise ValueError(f'{funcName}: block size of cunet model must be divisible by 4')

    if model == 0:
        if clip.format.id != vs.GRAYS:
            raise ValueError(f'{funcName}: input should be of GRAYS format')
    elif clip.format.id != vs.RGBS:
        raise ValueError(f'{funcName}: input should be of RGBS format')

    folder_path = os.path.join("waifu2x", tuple(Waifu2xModel.__members__)[model])

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

    width, height = clip.width, clip.height
    if model in (0, 1, 2):
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        clip = core.resize.Bicubic(
            clip,
            width * 2, height * 2,
            filter_param_a=0, filter_param_b=0.75
        )

    if backend == "ort-cpu":
        clip = core.ort.Model(
            clip, network_path,
            pad=pad, block_w=block_w, block_h=block_h,
            provider="CPU", builtin=1
        )
    elif backend == "ort-cuda":
        clip = core.ort.Model(
            clip, network_path,
            pad=pad, block_w=block_w, block_h=block_h,
            provider="CUDA", device_id=device_id, cudnn_benchmark=cudnn_benchmark,
            builtin=1
        )
    elif backend == "ov-cpu":
        clip = core.ov.Model(
            clip, network_path,
            pad=pad, block_w=block_w, block_h=block_h,
            device="CPU", builtin=1
        )
    else:
        raise ValueError(f'{funcName}: unknown backend {backend}: ["ort-cpu", "ort-cuda"]')

    if scale == 1 and clip.width // width == 2:
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        # cr: @AkarinVS
        clip = core.fmtc.resample(
            clip, scale=0.5,
            kernel="impulse", impulse=[-0.1875, 1.375, -0.1875],
            kovrspl=2
        )

    return clip