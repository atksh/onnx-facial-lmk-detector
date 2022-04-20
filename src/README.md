# Code

PyTorch codes of the pre- and post-processes and a script to merge and export to ONNX.

## Preliminalies

- docker

## How to run this code

```bash
docker build -t onnx-facial-lmk-detector .
docker run --rm -it -v $(pwd):/code onnx-facial-lmk-detector
```
