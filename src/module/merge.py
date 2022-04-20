import onnx
from onnx import helper, version_converter
from onnxsim import simplify

ONNX_OPSET_VERSION = 16
ONNX_AI_OPSET_VERSION = 3
ONNX_IR_VERSION = 8
OPSET_IMPORTS = [
    onnx.OperatorSetIdProto(domain="", version=ONNX_OPSET_VERSION),
    onnx.OperatorSetIdProto(domain="ai.onnx.ml", version=ONNX_AI_OPSET_VERSION),
]


def rename_dim_param(g, prefix):
    for value_info in g.value_info:
        try:
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    dim.dim_param = prefix + dim.dim_param
        except:
            pass


def make_tensor_value_info(name, x):
    t = x.type.tensor_type
    elem_type = t.elem_type
    shape = [s.dim_param if s.dim_param != "" else s.dim_value for s in t.shape.dim]
    return helper.make_tensor_value_info(name, elem_type, shape)


def match_versions(model1, model2):
    model1 = version_converter.convert_version(model1, ONNX_OPSET_VERSION)
    model2 = version_converter.convert_version(model2, ONNX_OPSET_VERSION)
    model1.ir_version = ONNX_IR_VERSION
    model2.ir_version = ONNX_IR_VERSION
    return model1, model2


def merge_models(model1, model2, io_map, output_names=None):
    model1, model2 = match_versions(model1, model2)
    input_names = [x.name for x in model1.graph.input]

    io_mapped_outputs = set([x[0] for x in io_map])
    _output_names = [
        x.name for x in model1.graph.output if x.name not in io_mapped_outputs
    ]
    _output_idxes = [
        (0, i)
        for i, x in enumerate(model1.graph.output)
        if x.name not in io_mapped_outputs
    ]
    num_model1_outputs = len(_output_names)
    _output_names += [x.name for x in model2.graph.output]
    _output_idxes += [(1, i) for i in range(len(model2.graph.output))]
    if output_names is None:
        output_names = _output_names
    elif len(output_names) != len(_output_names):
        raise ValueError

    model1 = onnx.compose.add_prefix(model1, "m1/")
    rename_dim_param(model1.graph, "m1/")
    model2 = onnx.compose.add_prefix(model2, "m2/")
    rename_dim_param(model2.graph, "m2/")
    io_map = [("m1/" + name1, "m2/" + name2) for (name1, name2) in io_map]

    model = onnx.compose.merge_models(model1, model2, io_map)

    def node_input(idx):
        return helper.make_node(
            "Identity",
            [input_names[idx]],
            [model1.graph.input[idx].name],
        )

    def node_output(idx):
        return helper.make_node(
            "Identity",
            [("m1/" if idx < num_model1_outputs else "m2/") + _output_names[idx]],
            [output_names[idx]],
        )

    graph_input = helper.make_graph(
        [node_input(i) for i in range(len(input_names))],
        "node_input",
        [
            make_tensor_value_info(name, x)
            for name, x in zip(input_names, model1.graph.input)
        ],
        [
            make_tensor_value_info(name, x)
            for name, x in zip([x.name for x in model1.graph.input], model1.graph.input)
        ],
    )
    current_output_names = [
        ("m1/" if idx < num_model1_outputs else "m2/") + _output_names[idx]
        for idx in range(len(output_names))
    ]

    outputs = [model1.graph.output, model2.graph.output]
    graph_outputs = [outputs[i][j] for i, j in _output_idxes]
    graph_output = helper.make_graph(
        [node_output(i) for i in range(len(output_names))],
        "node_output",
        [
            make_tensor_value_info(name, x)
            for name, x in zip(current_output_names, graph_outputs)
        ],
        [
            make_tensor_value_info(name, x)
            for name, x in zip(output_names, graph_outputs)
        ],
    )
    model_input = helper.make_model(
        graph_input, producer_name="model_input", opset_imports=OPSET_IMPORTS
    )
    model_output = helper.make_model(
        graph_output, producer_name="model_output", opset_imports=OPSET_IMPORTS
    )

    model, model_input = match_versions(model, model_input)
    model = onnx.compose.merge_models(
        model_input,
        model,
        io_map=[(x.name, x.name) for x in model_input.graph.output],
    )
    model, model_output = match_versions(model, model_output)
    model = onnx.compose.merge_models(
        model,
        model_output,
        io_map=[(x.name, x.name) for x in model_output.graph.input],
    )
    return model


def remove_initializer_from_input(model):
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if (
            initializer.name in name_to_input
            and name_to_input[initializer.name] in inputs
        ):
            inputs.remove(name_to_input[initializer.name])
    return model


def shape_inference(model):
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)


def optimize(model):
    model, check = simplify(
        model,
        dynamic_input_shape=True,
        input_shapes={"input": [640, 640, 3]},
        skipped_optimizers=[],
    )
    assert check
    remove_initializer_from_input(model)
    shape_inference(model)
    model, check = simplify(
        model,
        dynamic_input_shape=True,
        input_shapes={"input": [640, 640, 3]},
        skipped_optimizers=[],
    )
    assert check
    return model
