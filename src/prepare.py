from module.estimate_norm import *
from module.utils import *

if __name__ == "__main__":
    params = Params()
    m = Preprocess(params)
    m = torch.jit.script(m)

    dummy_input = torch.randn(640, 640, 3).to(dtype=torch.uint8)
    input_names = ["input"]
    output_names = ["output", "org_size", "org_img"]

    torch.onnx.export(
        m,
        dummy_input,
        "models/preprocess.onnx",
        export_params=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "h", 1: "w"},
            "org_img": {0: "h", 1: "w"},
        },
    )

    m = Forward(params)
    dummy_inputs = list()
    for dim in [1, 4, 10]:
        for stride in params.strides:
            size = (params.input_size // stride) ** 2 * params.num_anchors
            x = torch.ones((size, dim), dtype=torch.float32)
            dummy_inputs.append(x)

    input_names = [f"input.{i}" for i in range(len(dummy_inputs))]
    m = torch.jit.script(m)

    torch.onnx.export(
        m,
        dummy_inputs,
        "models/forward.onnx",
        export_params=True,
        opset_version=11,
        input_names=input_names,
        output_names=[
            "scores",
            "bboxes",
            "kpss",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
    )

    N = 10
    dummy_inputs = [
        torch.tensor([640, 640]),
        torch.randn(1, 1, 16800).float(),
        torch.randn(1, 16800, 4).float(),
        torch.randn(16800, 5, 2).float(),
        torch.arange(3 * N).view(N, 3).long(),
    ]

    m = PostProcess(params)
    m = torch.jit.script(m)

    torch.onnx.export(
        m,
        dummy_inputs,
        "models/postprocess.onnx",
        export_params=True,
        opset_version=11,
        input_names=["org_size", "scores.1", "bboxes.1", "kpss.1", "selected_idxes"],
        output_names=["scores", "bboxes", "kpss"],
        dynamic_axes={
            "selected_idxes": {0: "N"},
            "scores": {0: "N"},
            "bboxes": {0: "N"},
            "kpss": {0: "N"},
        },
    )

    dummy_inputs = [
        torch.randn(N).float(),
        torch.randn(N, 4).long(),
        torch.randn(N, 5, 2).long(),
        torch.randn(N, 224, 224, 3).to(dtype=torch.uint8),
        torch.randn(N, 106, 2).long(),
        torch.randn(N, 2, 3).float(),
    ]

    m = Output()
    m = torch.jit.script(m)

    torch.onnx.export(
        m,
        dummy_inputs,
        "models/output.onnx",
        export_params=True,
        opset_version=11,
        input_names=["scores.1", "bboxes.1", "kpss.1", "align_imgs.1", "lmks.1", "M.1"],
        output_names=["scores", "bboxes", "kpss", "align_imgs", "lmks", "M"],
        dynamic_axes={
            "scores.1": {0: "M"},
            "scores": {0: "N"},
            "bboxes.1": {0: "M"},
            "bboxes": {0: "N"},
            "kpss.1": {0: "M"},
            "kpss": {0: "N"},
            "align_imgs.1": {0: "M"},
            "align_imgs": {0: "N"},
            "lmks.1": {0: "M"},
            "lmks": {0: "N"},
            "M.1": {0: "M"},
            "M": {0: "N"},
        },
    )

    post_lmk = PostLandmark()
    IM = torch.randn(1, 2, 3).float()
    lmks = torch.randn(1, 106 * 2).float()
    post_lmk = torch.jit.script(post_lmk)

    torch.onnx.export(
        post_lmk,
        [IM, lmks],
        "models/post_landmarks.onnx",
        export_params=True,
        opset_version=12,
        input_names=["IM.input", "lmks"],
        output_names=["landmarks"],
        dynamic_axes={
            "IM.input": {0: "N"},
            "lmks": {0: "N"},
            "landmarks": {0: "N"},
        },
    )

    img = (torch.rand((640, 640, 3)) * 255).to(dtype=torch.uint8)
    kpss = torch.tensor([[[124, 48], [170, 44], [117, 76], [133, 94], [170, 90]]])
    kpss = kpss.repeat((4, 1, 1))

    m = EstimateNorm()
    y = m(kpss, img)

    m = torch.jit.script(m)
    torch.onnx.export(
        m,
        [kpss, img],
        "models/estimate_norm.onnx",
        export_params=True,
        opset_version=12,
        input_names=["kpss.en", "img"],
        output_names=["keypoints", "IM", "align_imgs", "tensor_imgs", "M"],
        dynamic_axes={
            "img": {0: "h", 1: "w"},
            "kpss.en": {0: "N"},
            "M": {0: "N"},
            "IM": {0: "N"},
            "keypoints": {0: "N"},
            "align_imgs": {0: "N", 1: "h_out", 2: "w_out"},
            "tensor_imgs": {0: "N", 2: "h_out2", 3: "w_out2"},
        },
    )
