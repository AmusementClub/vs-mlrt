#ifndef CONVERT_FLOAT_TO_FLOAT16_H
#define CONVERT_FLOAT_TO_FLOAT16_H

#include <string>
#include <unordered_set>

#include <onnx/onnx_pb.h>

void convert_float_to_float16(
    ONNX_NAMESPACE::ModelProto & model,
    bool force_fp16_initializers,
    // bool keep_io_types = True,
    // bool disable_shape_infer = True,
    // const std::optional<std::unordered_set<std::string>> op_block_list = DEFAULT_OP_BLOCK_LIST,
    // const std::optional<std::unordered_set<std::string>> op_block_list = {},
    const std::unordered_set<std::string> & op_block_list,
    bool cast_input = true,
    bool cast_output = true
) noexcept;

#endif
