import onnx
from onnx import helper

from module.merge import (
    OPSET_IMPORTS,
    make_tensor_value_info,
    merge_models,
    optimize,
    remove_initializer_from_input,
)


def get_nms(forward):
    node = helper.make_node(
        "NonMaxSuppression",
        inputs=[
            "bboxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        outputs=["selected_idxes"],
    )

    def node_input(name):
        return helper.make_node("Identity", [name], [f"{name}.out"])

    selected_idxes = helper.make_tensor_value_info(
        "selected_idxes",
        onnx.TensorProto.INT64,
        ["N", 3],
    )
    graph = helper.make_graph(
        [node, node_input("bboxes"), node_input("scores")],
        "nms",
        [
            make_tensor_value_info(name, x)
            for name, x in zip(
                [f"{x.name}" for x in forward.graph.output if x.name != "kpss"],
                [x for x in forward.graph.output if x.name != "kpss"],
            )
        ],
        [
            selected_idxes,
            make_tensor_value_info("scores.out", forward.graph.output[0]),
            make_tensor_value_info("bboxes.out", forward.graph.output[1]),
        ],
    )
    model = helper.make_model(graph, producer_name="nms", opset_imports=OPSET_IMPORTS)
    model = onnx.compose.add_prefix(model, "nms/")
    return model


if __name__ == "__main__":
    detect_model = onnx.load("models/scrfd_10g_bnkps.onnx")
    landmark_model = onnx.load("models/2d106det.onnx")

    preprocess = onnx.load("models/preprocess.onnx")
    forward = onnx.load("models/forward.onnx")
    postprocess = onnx.load("models/postprocess.onnx")
    estimate_norm = onnx.load("models/estimate_norm.onnx")
    post_landmarks = onnx.load("models/post_landmarks.onnx")
    output = onnx.load("models/output.onnx")
    nms = get_nms(forward)

    forward = merge_models(
        forward,
        nms,
        io_map=[
            (x.name, f"nms/{x.name}") for x in forward.graph.output if x.name != "kpss"
        ],
    )

    model_forward = merge_models(
        detect_model,
        forward,
        io_map=[
            (x.name, f"input.{i}") for i, x in enumerate(detect_model.graph.output)
        ],
    )
    merged_model = merge_models(
        preprocess, model_forward, io_map=[("output", "input.1")]
    )
    merged_model = merge_models(
        merged_model,
        postprocess,
        io_map=[
            ("org_size", "org_size"),
            ("nms/scores.out", "scores.1"),
            ("nms/bboxes.out", "bboxes.1"),
            ("kpss", "kpss.1"),
            ("nms/selected_idxes", "selected_idxes"),
        ],
    )
    merged_model = merge_models(
        merged_model,
        estimate_norm,
        io_map=[("kpss", "kpss.en"), ("org_img", "img")],
    )
    merged_model = merge_models(
        merged_model,
        landmark_model,
        io_map=[("tensor_imgs", "data")],
    )

    output_names = ["scores", "bboxes", "kpss", "align_imgs", "M", "lmks"]
    merged_model = merge_models(
        merged_model,
        post_landmarks,
        io_map=[("IM", "IM.input"), ("fc1", "lmks")],
        output_names=output_names,
    )
    output_names = ["scores", "bboxes", "kpss", "align_imgs", "lmks", "M"]
    merged_model = merge_models(
        merged_model,
        output,
        io_map=[(n, f"{n}.1") for n in output_names],
        output_names=output_names,
    )

    merged_model = remove_initializer_from_input(merged_model)
    merged_model = optimize(merged_model)
    while len(merged_model.opset_import) > 1:
        merged_model.opset_import.remove(merged_model.opset_import[-1])

    onnx.save(merged_model, "model.onnx")
