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
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

# 1. Load pretrained ResNet50 (ImageNet weights)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# 2. Create dummy input (batch_size=1, 3x224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
onnx_path = "resnet50.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"Exported ONNX model to {onnx_path}")