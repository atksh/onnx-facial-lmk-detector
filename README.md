# onnx-facial-lmk-detector
End-to-end face detection, cropping, norm estimation, and landmark detection in a **single onnx model**, `model.onnx`.

## Example

![example](https://raw.githubusercontent.com/atksh/onnx-facial-lmk-detector/6ea090532acce1c228d1f860d27708d450416475/output.png?token=GHSAT0AAAAAABHJHGPX4XIAJZ4ALEVWPJTIYSJ6HKQ)


```python
import onnxruntime as ort
import cv2

sess = ort.InferenceSession("model.onnx")


img = cv2.imread("sample.jpg")

scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = sess.run(None, {"input": img})
```