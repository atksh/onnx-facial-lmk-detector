import enum
from asyncio import transports

import cv2
import numpy as np
import onnxruntime as ort
import torch

EP_list = ["CPUExecutionProvider"]
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess = ort.InferenceSession("model.onnx", sess_options, providers=EP_list)


def crop_faces(image_full: np.ndarray) -> np.ndarray:
    outputs = [
        "scores",
        "bboxes",
        "kpss",
        "align_imgs",
        "lmks",
        "M",
    ]
    scores, bboxes, kpss, imgs, lmks, M = sess.run(outputs, {"input": image_full})
    print(scores.shape, bboxes.shape, kpss.shape, imgs.shape, lmks.shape, M.shape)
    print(scores.dtype, bboxes.dtype, kpss.dtype, imgs.dtype, lmks.dtype, M.dtype)
    return bboxes, kpss, imgs, lmks, M


img = cv2.imread("sample.jpg")
org_img = img.copy()
org_img_copy = img.copy()
bboxes, kpss, imgs, lmks, Ms = crop_faces(img)
for idx, (bbox, kps, img, lmk, M) in enumerate(zip(bboxes, kpss, imgs, lmks, Ms)):
    color = (200, 160, 75)
    pt1, pt2 = bbox[:2], bbox[2:]
    cv2.rectangle(org_img, pt1, pt2, color=(0, 200, 0), thickness=3)

    for x, y in lmk:
        x = int(round(x))
        y = int(round(y))
        cv2.circle(org_img, (x, y), 0, color, 3)

    cv2.imwrite(f"crops/{idx}.jpg", img)

cv2.imwrite(f"sample_out.png", org_img)

