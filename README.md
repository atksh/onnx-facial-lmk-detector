# onnx-facial-lmk-detector

End-to-end face detection, cropping, norm estimation, and landmark detection in a **single onnx model**, `model.onnx`.

### Demo

You can try this model at the following link. Thanks for hysts.

- https://huggingface.co/spaces/hysts/atksh-onnx-facial-lmk-detector

### Code

See [src](/src/README.md).

## Example

![example](https://raw.githubusercontent.com/atksh/onnx-facial-lmk-detector/6ea090532acce1c228d1f860d27708d450416475/output.png?token=GHSAT0AAAAAABHJHGPX4XIAJZ4ALEVWPJTIYSJ6HKQ)

```python
import onnxruntime as ort
import cv2

sess = ort.InferenceSession("model.onnx")
img = cv2.imread("input.jpg")

scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = sess.run(None, {"input": img})
# float32 int64 int64 uint8 int64 float32
# (N,) (N, 4) (N, 5, 2) (N, 224, 224, 3) (N, 106, 2) (N, 2, 3)
```

This model requires `onnxruntime>=1.11`.

## How does it work?

This is simply a merged model of the following underlying models with some pre- and post-processing.

### Underlying models

|                    | model         | reference                                                                                                   |
| ------------------ | ------------- | ----------------------------------------------------------------------------------------------------------- |
| face detection     | SCRFD_10G_KPS | https://github.com/deepinsight/insightface/tree/master/detection/scrfd#pretrained-models                    |
| landmark detection | 2d106det      | https://github.com/deepinsight/insightface/blob/master/alignment/coordinate_reg/README.md#pretrained-models |

### Pre- and Post-Processing

Implemented the following processing by PyTorch and exported to ONNX.

- Input transform:

  - Resize and pad to (1920, 1920)
  - BGR to RGB conversion
  - Transpose (H, W, C) to (C, H, W)

- (Face Detection)
- Post-processing of face detection

  - Predicted bounding boxes and Confidence Score Processing
  - NMS (ONNX Operator)

- Norm estimation and face cropping

  - Estimate the norm and apply an affine transformation to each face.
  - Crop the faces and resize them to (192, 192).

- (Landmark Detection)
- Perform post-processing for landmark detection.

  - Process the predicted landmarks and apply the inverse affine transform to each face.

## Note

Please check with the model provider regarding the license for your use.

This model includes the work that is distributed in the Apache License 2.0.
