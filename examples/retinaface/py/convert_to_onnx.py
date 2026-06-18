# -*- coding: utf-8 -*-

"""
Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from modelscope.pipelines.cv.retina_face_detection_pipeline import RetinaFaceDetection

model_path = "./pytorch_model.pt"
onnx_path = "retinaface_resnet50.onnx"

model = RetinaFaceDetection(model_path, device="cpu")
net = model.net
net.eval()

dummy = torch.randn(1, 3, 640, 640, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        opset_version=11,
        do_constant_folding=True
    )

print("saved:", onnx_path)
