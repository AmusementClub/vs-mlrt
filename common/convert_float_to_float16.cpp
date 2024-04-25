// re-implemented and modified from
// https://github.com/microsoft/onnxruntime/blob/v1.10.0/onnxruntime/python/tools/transformers/float16.py

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <onnx/onnx_pb.h>

#include "convert_float_to_float16.h"


namespace {
struct InitializerTracker {
    ONNX_NAMESPACE::TensorProto * initializer;
    std::vector<ONNX_NAMESPACE::NodeProto *> fp32_nodes;
    std::vector<ONNX_NAMESPACE::NodeProto *> fp16_nodes;

    InitializerTracker(ONNX_NAMESPACE::TensorProto * initializer) :
        initializer(initializer), fp32_nodes(), fp16_nodes() {}

    void add_node(
        ONNX_NAMESPACE::NodeProto * node,
        bool is_node_blocked
    ) noexcept {

        if (is_node_blocked) {
            fp32_nodes.emplace_back(node);
        } else {
            fp16_nodes.emplace_back(node);
        }
    }
};
}


template <typename T>
static ONNX_NAMESPACE::AttributeProto make_attribute(
    const std::string_view & key,
    const T & value
) noexcept {

    auto attr = ONNX_NAMESPACE::AttributeProto{};

    attr.set_name(std::data(key), std::size(key));

    if constexpr (std::is_same_v<T, float>) {
        attr.set_f(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    } else if constexpr (std::is_same_v<T, int>) {
        attr.set_i(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    } else if constexpr (std::is_same_v<std::underlying_type_t<T>, int>) {
        attr.set_i(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    } else if constexpr (std::is_same_v<T, std::string>) {
        attr.set_s(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    } else if constexpr (std::is_same_v<T, ONNX_NAMESPACE::TensorProto>) {
        *attr.mutable_t() = value;
        attr.set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
    } else if constexpr (std::is_same_v<T, ONNX_NAMESPACE::SparseTensorProto>) {
        *attr.mutable_sparse_tensor() = value;
        attr.set_type(ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR);
    } else if constexpr (std::is_same_v<T, ONNX_NAMESPACE::GraphProto>) {
        *attr.mutable_g() = value;
        attr.set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
    } else if constexpr (std::is_same_v<T, ONNX_NAMESPACE::TypeProto>) {
        *attr.mutable_tp() = value;
        attr.set_type(ONNX_NAMESPACE::AttributeProto::TYPE_PROTO);
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
        attr.mutable_ints()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        attr.mutable_floats()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOATS);
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        attr.mutable_strings()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);
    } else if constexpr (std::is_same_v<T, std::vector<ONNX_NAMESPACE::TensorProto>>) {
        attr.mutable_tensors()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::TENSORS);
    } else if constexpr (std::is_same_v<T, std::vector<ONNX_NAMESPACE::SparseTensorProto>>) {
        attr.mutable_sparse_tensors()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS);
    } else if constexpr (std::is_same_v<T, std::vector<ONNX_NAMESPACE::GraphProto>>) {
        attr.mutable_graphs()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::GRAPHS);
    } else if constexpr (std::is_same_v<T, std::vector<ONNX_NAMESPACE::TypeProto>>) {
        attr.mutable_type_protos()->CopyFrom(value);
        attr.set_type(ONNX_NAMESPACE::AttributeProto::TYPE_PROTOS);
    } else {
        abort();
    }

    return attr;
}


// base
static inline ONNX_NAMESPACE::NodeProto make_node(
    ONNX_NAMESPACE::NodeProto && node
) noexcept {
    return node;
}

// recursive
template <typename T, typename... Targs>
static inline ONNX_NAMESPACE::NodeProto make_node(
    ONNX_NAMESPACE::NodeProto && node,
    const std::string_view & key,
    const T & value,
    Targs... kwargs
) noexcept {

    node.mutable_attribute()->Add(make_attribute(key, value));
    return make_node(std::move(node), kwargs...);
}

// frontend
template <typename... Targs>
static ONNX_NAMESPACE::NodeProto make_node(
    const std::string_view & op_type,
    const std::vector<std::string> & inputs,
    const std::vector<std::string> & outputs,
    const std::string_view & name,
    Targs... kwargs
) noexcept {

    auto node = ONNX_NAMESPACE::NodeProto{};

    node.set_op_type(std::data(op_type), std::size(op_type));
    node.mutable_input()->CopyFrom({std::cbegin(inputs), std::cend(inputs)});
    node.mutable_output()->CopyFrom({std::cbegin(outputs), std::cend(outputs)});

    node.set_name(std::data(name), std::size(name));

    static_assert(sizeof...(kwargs) % 2 == 0, "format: key1, value1, ...");
    if constexpr (constexpr auto size = sizeof...(kwargs) / 2; size > 0) {
        node.mutable_attribute()->Reserve(sizeof...(kwargs) / 2);
    }

    return make_node(std::move(node), kwargs...);
}


// simplified from
// https://github.com/numpy/numpy/blob/v1.21.5/numpy/core/src/npymath/halffloat.c#L243-L364
// Inf or NaN overflow to signed inf
// underflow to signed zero
// NPY_HALF_ROUND_TIES_TO_EVEN = 1
// NPY_HALF_GENERATE_OVERFLOW = 0
// NPY_HALF_GENERATE_UNDERFLOW 0
static inline uint16_t float_to_half(uint32_t f) noexcept {
    uint16_t h_sgn = static_cast<uint16_t>((f & 0x80000000u) >> 16);
    uint32_t f_exp = f & 0x7f800000u;

    /* Exponent overflow/NaN converts to signed inf */
    if (f_exp >= 0x47800000u) {
        return static_cast<uint16_t>(h_sgn + 0x7c00u);
    }

    /* Exponent underflow converts to a signed zero */
    if (f_exp <= 0x38000000u) {
        return h_sgn;
    }

    uint16_t h_exp = static_cast<uint16_t>((f_exp - 0x38000000u) >> 13);
    uint32_t f_sig = f & 0x007fffffu;
    if ((f_sig & 0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
    uint16_t h_sig = static_cast<uint16_t>(f_sig >> 13);
    return h_sgn + h_exp + h_sig;
}


template <typename T>
static inline void convert_float_to_float16(
    T * __restrict dst_array,
    const float * __restrict src_array,
    int n
) noexcept {

    auto src_array_u32 = reinterpret_cast<const uint32_t *>(src_array);

    for (int i = 0; i < n; ++i) {
        dst_array[i] = static_cast<T>(float_to_half(src_array_u32[i]));
    }
}


static void convert_tensor_float_to_float16(
    ONNX_NAMESPACE::TensorProto & tensor
) noexcept {

    if (tensor.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
        tensor.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT16);
        // convert float_data (float type) to float16 and write to int32_data
        if (tensor.float_data_size() != 0) {
            int n = tensor.float_data_size();
            tensor.mutable_int32_data()->Resize(n, 0);
            convert_float_to_float16(
                tensor.mutable_int32_data()->mutable_data(),
                std::data(tensor.float_data()),
                n
            );
            tensor.clear_float_data();
        }
        if (tensor.has_raw_data()) {
            std::string & raw_data = *tensor.mutable_raw_data();

            auto nbytes = std::size(raw_data);
            std::unique_ptr<float [], decltype(&free)> data {
                (float *) malloc(nbytes),
                free
            };
            memcpy(data.get(), std::data(raw_data), nbytes);

            auto n = nbytes / sizeof(float);
            raw_data.resize(n * sizeof(uint16_t));

            convert_float_to_float16((uint16_t *) std::data(raw_data), data.get(), n);
        }
    }
}


static ONNX_NAMESPACE::TypeProto make_tensor_type_proto(
    int32_t elem_type,
    const google::protobuf::RepeatedField<google::protobuf::int64> & shape
) noexcept {

    auto type_proto = ONNX_NAMESPACE::TypeProto{};

    auto & tensor_type_proto = *type_proto.mutable_tensor_type();
    tensor_type_proto.set_elem_type(elem_type);

    auto & tensor_shape_proto = *tensor_type_proto.mutable_shape();
    tensor_shape_proto.mutable_dim()->Reserve(shape.size());
    for (auto d : shape) {
        auto & dim = *tensor_shape_proto.mutable_dim()->Add();
        dim.set_dim_value(d);
    }

    return type_proto;
}


static ONNX_NAMESPACE::ValueInfoProto make_tensor_value_info(
    const std::string_view & name,
    int32_t elem_type,
    const google::protobuf::RepeatedField<google::protobuf::int64> & shape,
    const std::optional<std::string> & doc_string = {}
) noexcept {

    auto value_info_proto = ONNX_NAMESPACE::ValueInfoProto{};

    value_info_proto.set_name(std::data(name), std::size(name));

    if (doc_string.has_value()) {
        value_info_proto.set_doc_string(doc_string.value());
    }

    *value_info_proto.mutable_type() = make_tensor_type_proto(elem_type, shape);

    return value_info_proto;
}


static ONNX_NAMESPACE::ValueInfoProto make_value_info_from_tensor(
    const ONNX_NAMESPACE::TensorProto & tensor
) noexcept {

    return make_tensor_value_info(
        tensor.name(),
        tensor.data_type(),
        tensor.dims()
    );
}


void convert_float_to_float16(
    ONNX_NAMESPACE::ModelProto & model,
    bool force_fp16_initializers,
    // bool keep_io_types = True,
    // bool disable_shape_infer = True,
    // const std::optional<std::unordered_set<std::string>> op_block_list = DEFAULT_OP_BLOCK_LIST,
    // const std::optional<std::unordered_set<std::string>> op_block_list = {},
    const std::unordered_set<std::string> & op_block_list,
    bool cast_input,
    bool cast_output
) noexcept {

    std::vector<ONNX_NAMESPACE::ValueInfoProto> value_info_list {};
    std::unordered_set<std::string> io_casts {};

    std::unordered_map<std::string, std::string> name_mapping {};
    std::unordered_set<std::string> graph_io_to_skip {};

    if (cast_input) {
        const std::vector<std::string> fp32_inputs = [&]() {
            std::vector<std::string> ret {};

            for (const auto & n : model.graph().input()) {
                if (n.type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
                    ret.emplace_back(n.name());
                }
            }

            return ret;
        }();

        for (const auto & n : model.graph().input()) {
            if (auto idx = std::find(std::cbegin(fp32_inputs), std::cend(fp32_inputs), n.name());
                idx != std::cend(fp32_inputs)
            ) {
                const auto i = idx - std::cbegin(fp32_inputs);
                std::string node_name = "graph_input_cast_" + std::to_string(i);
                name_mapping.emplace(n.name(), node_name);
                graph_io_to_skip.emplace(n.name());

                auto * new_value_info = model.mutable_graph()->mutable_value_info()->Add();
                new_value_info->CopyFrom(n);
                new_value_info->set_name(node_name);
                new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
                    ONNX_NAMESPACE::TensorProto::FLOAT16
                );
                // add Cast node (from tensor(float) to tensor(float16) after graph input
                for (auto & node : *model.mutable_graph()->mutable_node()) {
                    for (auto & input : *node.mutable_input()) {
                        if (input == n.name()) {
                            input = node_name;
                        }
                    }
                }
                auto new_node = make_node(
                    "Cast", {n.name()}, {node_name}, node_name,
                    "to", ONNX_NAMESPACE::TensorProto::FLOAT16
                );
                model.mutable_graph()->mutable_node()->Add();
                for (int i = model.graph().node_size() - 2; i >= 0; --i) {
                    model.mutable_graph()->mutable_node()->SwapElements(i, i + 1);
                }
                *model.mutable_graph()->mutable_node(0) = std::move(new_node);
                value_info_list.emplace_back(*new_value_info);
                io_casts.emplace(std::move(node_name));
            }
        }
    }

    if (cast_output) {
        const std::vector<std::string> fp32_outputs = [&]() {
            std::vector<std::string> ret {};

            for (const auto & n : model.graph().output()) {
                if (n.type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
                    ret.emplace_back(n.name());
                }
            }

            return ret;
        }();

        for (const auto & n : model.graph().output()) {
            if (auto idx = std::find(std::cbegin(fp32_outputs), std::cend(fp32_outputs), n.name());
                idx != std::cend(fp32_outputs)
            ) {
                const auto i = idx - std::cbegin(fp32_outputs);
                std::string node_name = "graph_output_cast_" + std::to_string(i);
                name_mapping.emplace(n.name(), node_name);
                graph_io_to_skip.emplace(n.name());

                auto * new_value_info = model.mutable_graph()->mutable_value_info()->Add();
                new_value_info->CopyFrom(n);
                new_value_info->set_name(node_name);
                new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
                    ONNX_NAMESPACE::TensorProto::FLOAT16
                );
                // add Cast node (from tensor(float16) to tensor(float) before graph output
                for (auto & node : *model.mutable_graph()->mutable_node()) {
                    for (auto & output : *node.mutable_output()) {
                        if (output == n.name()) {
                            output = node_name;
                        }
                    }
                }
                auto new_node = make_node(
                    "Cast", {node_name}, {n.name()}, node_name,
                    "to", ONNX_NAMESPACE::TensorProto::FLOAT
                );
                model.mutable_graph()->mutable_node()->Add(std::move(new_node));
                value_info_list.emplace_back(*new_value_info);
                io_casts.emplace(std::move(node_name));
            }
        }
    }

    std::vector<ONNX_NAMESPACE::NodeProto *> node_list {};

    std::vector<std::variant<
        ONNX_NAMESPACE::ModelProto *,
        ONNX_NAMESPACE::GraphProto *,
        ONNX_NAMESPACE::AttributeProto *
    >> queue {};
    queue.emplace_back(&model);

    std::unordered_map<std::string, InitializerTracker> fp32_initializers {};
    while (!std::empty(queue)) {
        decltype(queue) next_level {};
        for (auto & q : queue) {
            // if q is model, push q.graph (GraphProto)
            if (std::holds_alternative<ONNX_NAMESPACE::ModelProto *>(q)) {
                next_level.emplace_back(
                    std::get<ONNX_NAMESPACE::ModelProto *>(q)->mutable_graph()
                );

            // if q is model.graph, push q.node.attribute (AttributeProto)
            } else if (std::holds_alternative<ONNX_NAMESPACE::GraphProto *>(q)) {
                auto * q_ = std::get<ONNX_NAMESPACE::GraphProto *>(q);

                for (auto & n : *q_->mutable_initializer()) {
                    if (n.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
                        assert(fp32_initializers.count(n.name()) == 0);
                        fp32_initializers.emplace(n.name(), InitializerTracker(&n));
                    }
                }

                for (auto & n : *q_->mutable_node()) {
                    // if n is in the block list (doesn't support float16), no conversion for the node,
                    // and save the node for further processing
                    if (io_casts.count(n.name()) != 0) {
                        continue;
                    }

                    bool is_node_blocked = op_block_list.count(n.op_type()) != 0;

                    for (auto & input : *n.mutable_input()) {
                        if (auto idx = name_mapping.find(input);
                            idx != std::cend(name_mapping)
                        ) {
                            input = idx->second;
                        }

                        if (auto idx = fp32_initializers.find(input);
                            idx != std::cend(fp32_initializers)
                        ) {
                            idx->second.add_node(&n, is_node_blocked);
                        }
                    }
                    for (auto & output : *n.mutable_output()) {
                        if (auto idx = name_mapping.find(output);
                            idx != std::cend(name_mapping)
                        ) {
                            output = idx->second;
                        }
                    }

                    if (is_node_blocked) {
                        node_list.emplace_back(&n);
                    } else {
                        if (n.op_type() == "Cast") {
                            for (auto & attr : *n.mutable_attribute()) {
                                if (attr.name() == "to" &&
                                    attr.i() == ONNX_NAMESPACE::TensorProto::FLOAT
                                ) {
                                    attr.set_i(ONNX_NAMESPACE::TensorProto::FLOAT16);
                                    break;
                                }
                            }
                        }

                        if (n.attribute_size() != 0) {
                            for (auto & attr : *n.mutable_attribute()) {
                                next_level.emplace_back(&attr);
                            }
                        }
                    }
                }

                // if q is graph, process input, output and value_info (ValueInfoProto)
                const auto func = [&](auto & n) {
                    if (n.type().tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
                        if (graph_io_to_skip.count(n.name()) == 0) {
                            n.mutable_type()->mutable_tensor_type()->set_elem_type(
                                ONNX_NAMESPACE::TensorProto::FLOAT16
                            );
                            value_info_list.emplace_back(n);
                        }
                    }
                };

                for (auto & n : *q_->mutable_input()) {
                    func(n);
                }

                for (auto & n : *q_->mutable_output()) {
                    func(n);
                }

                for (auto & n : *q_->mutable_value_info()) {
                    func(n);
                }

            // if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            // and process node.attribute.t and node.attribute.tensors (TensorProto)
            } else if (std::holds_alternative<ONNX_NAMESPACE::AttributeProto *>(q)) {
                auto & q_ = std::get<ONNX_NAMESPACE::AttributeProto *>(q);
                if (q_->has_g()) {
                    next_level.emplace_back(q_->mutable_g());
                }
                if (q_->graphs_size() != 0) {
                    for (auto & graph : *q_->mutable_graphs()) {
                        next_level.emplace_back(&graph);
                    }
                }
                if (q_->has_t()) {
                    convert_tensor_float_to_float16(*q_->mutable_t());
                }
                if (q_->tensors_size() != 0) {
                    for (auto & n : *q_->mutable_tensors()) {
                        convert_tensor_float_to_float16(n);
                    }
                }
            }
        }

        queue = std::move(next_level);
    }

    for (auto & [_, value] : fp32_initializers) {
        // to avoid precision loss,
        // do not convert an initializer to fp16 when it is used only by fp32 nodes.
        if (force_fp16_initializers || !std::empty(value.fp16_nodes)) {
            convert_tensor_float_to_float16(*value.initializer);
            value_info_list.emplace_back(make_value_info_from_tensor(*value.initializer));
            if (!std::empty(value.fp32_nodes) && !force_fp16_initializers) {
                std::cerr << "initializer is used by both fp32 and fp16 nodes.";
            }
        }
    }

    // process the nodes in block list that doesn't support tensor(float16)
    for (auto & node : node_list) {
        // if input's name is in the value_info_list meaning input is tensor(float16) type,
        // insert a float16 to float Cast node before the node,
        // change current node's input name and create new value_info for the new name
        for (int i = 0; i < node->input_size(); ++i) {
            auto & input = *node->mutable_input(i);
            for (const auto & value_info : value_info_list) {
                if (input == value_info.name()) {
                    // create new value_info for current node's new input name
                    auto * new_value_info = model.mutable_graph()->mutable_value_info()->Add();
                    new_value_info->CopyFrom(value_info);
                    std::string output_name = node->name() + "_input_cast_" + std::to_string(i);
                    new_value_info->set_name(output_name);
                    new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
                        ONNX_NAMESPACE::TensorProto::FLOAT
                    );
                    // add Cast node (from tensor(float16) to tensor(float) before current node
                    std::string node_name = node->name() + "_input_cast" + std::to_string(i);
                    auto new_node = make_node(
                        "Cast", {input}, {output_name}, node_name,
                        "to", ONNX_NAMESPACE::TensorProto::FLOAT
                    );
                    model.mutable_graph()->mutable_node()->Add(std::move(new_node));
                    input = std::move(output_name);
                    break;
                }
            }
        }
        // if output's name is in the value_info_list meaning output is tensor(float16) type,
        // insert a float to float16 Cast node after the node,
        // change current node's output name and create new value_info for the new name
        for (int i = 0; i < node->output_size(); ++i) {
            auto & output = *node->mutable_output(i);
            for (const auto & value_info : value_info_list) {
                if (output == value_info.name()) {
                    // create new value_info for current node's new output
                    auto * new_value_info = model.mutable_graph()->mutable_value_info()->Add();
                    new_value_info->CopyFrom(value_info);
                    std::string input_name = node->name() + "_output_cast_" + std::to_string(i);
                    new_value_info->set_name(input_name);
                    new_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
                            ONNX_NAMESPACE::TensorProto::FLOAT
                        );
                    // add Cast node (from tensor(float) to tensor(float16) after current node
                    const std::string node_name = node->name() + "_output_cast" + std::to_string(i);
                    auto new_node = make_node(
                        "Cast", {input_name}, {output}, node_name,
                        "to", ONNX_NAMESPACE::TensorProto::FLOAT16
                    );
                    model.mutable_graph()->mutable_node()->Add(std::move(new_node));
                    output = std::move(input_name);
                    break;
                }
            }
        }
    }
}
