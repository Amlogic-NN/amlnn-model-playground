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
import os
import sys
import torch
import warnings
import onnx
from onnxsim import simplify
from modelscope.hub.snapshot_download import snapshot_download

warnings.filterwarnings('ignore')

# 1. Configuration
MODELSCOPE_ID = "IoT-Edge/Gesture_Detect" 
ONNX_PATH = "gesture.onnx"

print(f"1. Downloading model repository from ModelScope ({MODELSCOPE_ID})...")
model_dir = snapshot_download(MODELSCOPE_ID)
sys.path.insert(0, model_dir)

# 2. Load the model
pt_path = os.path.join(model_dir, "best.pt")
print(f"2. Loading PyTorch weights from {pt_path}...")

checkpoint = torch.load(pt_path, map_location="cpu")
if isinstance(checkpoint, dict):
    model = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
else:
    model = checkpoint

model.float().eval()

# =====================================================================
# Bypass the Complex Detect Head
# Instead of doing bounding box math, force the model to just output
# the raw 4D feature maps
# =====================================================================
def bypass_forward(self, x):
    # Simply return the input feature maps, skipping all anchor processing
    return x

for m in model.modules():
    if type(m).__name__ in ["Detect", "Segment", "Pose", "v10Detect"]:
        # Monkey-patch the forward function
        m.forward = bypass_forward.__get__(m)

# 3. Export to ONNX
print(f"3. Exporting to {ONNX_PATH}...")
dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0", "output1", "output2"] # Usually 3 raw heads
    )

print("4. Running ONNX Simplifier to fix the graph architecture...")

try:
    onnx_model = onnx.load(ONNX_PATH)
    # Simplify the model
    model_simp, check = simplify(onnx_model)
    
    if check:
        onnx.save(model_simp, ONNX_PATH)
        print(f"Success! Simplified ONNX model saved to {ONNX_PATH}")
    else:
        print("Warning: ONNX Simplifier failed to validate the graph, but saved anyway.")
        onnx.save(model_simp, ONNX_PATH)
except Exception as e:
    print(f"ONNX Simplifier encountered an error: {e}")

print("Export Complete! Try converting this new model with ADLA.")