#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
We use
https://hf-mirror.com/yuekai/model_repo_sense_voice_small/blob/main/export_onnx.py
as a reference while writing this file.

Thanks to https://github.com/yuekaizhang for making the file public.
"""

import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # absolute path of this file
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # absolute path of the parent directory
sys.path.append(parent_dir)  # add the parent directory to sys.path

from typing import Any, Dict, Tuple

OUTPUT_DIR = os.path.join(current_dir, "onnx-3input")

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def apply_pytorch_module_compat_patch() -> None:
    """Work around model.py not calling super().__init__() in SinusoidalPositionEncoder.

    PyTorch 2.12+ expects nn.Module subclasses to be fully initialized before
    state_dict() is called during pretrained weight loading.
    """
    if getattr(apply_pytorch_module_compat_patch, "_applied", False):
        return
    apply_pytorch_module_compat_patch._applied = True

    _orig_state_dict = torch.nn.Module.state_dict

    def state_dict(self, *args, **kwargs):
        if not hasattr(self, "_state_dict_pre_hooks"):
            torch.nn.Module.__init__(self)
        return _orig_state_dict(self, *args, **kwargs)

    torch.nn.Module.state_dict = state_dict


def load_model(model_id: str = "iic/SenseVoiceSmall"):
    from funasr import AutoModel

    apply_pytorch_module_compat_patch()
    return AutoModel.build_model(
        model=model_id,
        trust_remote_code=True,
        remote_code=os.path.join(current_dir, "model.py"),
        device="cpu",
    )


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def modified_forward(
    self,
    x: torch.Tensor,
    language: torch.Tensor,
    text_norm: torch.Tensor,
):
    """
    Args:
      x: A 3-D tensor of shape (batch_size, sequence_length, feature_dim) with dtype torch.float32.
    """
    device = x.device
    x_length = torch.full(
        (x.size(0),), x.shape[1], dtype=torch.int32, device=device
    )

    language_query = self.embed(language.to(device)).unsqueeze(1)
    textnorm_query = self.embed(text_norm.to(device)).unsqueeze(1)

    x = torch.cat((textnorm_query, x), dim=1)
    x_length = x_length + 1

    event_emo_query = self.embed(
        torch.tensor([[1, 2]], dtype=torch.long, device=device)
    ).repeat(x.size(0), 1, 1)
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    x = torch.cat((input_query, x), dim=1)
    x_length = x_length + 3

    encoder_out, encoder_out_lens = self.encoder(x, x_length)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    # Output CTC logits
    ctc_logits = self.ctc.ctc_lo(encoder_out)

    return ctc_logits


def load_cmvn(filename) -> Tuple[str, str]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def generate_tokens(params, output_dir: str):
    sp = params["tokenizer"].sp
    tokens_file = os.path.join(output_dir, "tokens.txt")
    with open(tokens_file, "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

    os.system(f"head {tokens_file}; tail -n200 {tokens_file}")


def display_params(params):
    print("----------params----------")
    print(params)

    print("----------frontend_conf----------")
    print(params["frontend_conf"])

    os.system(f"cat {params['frontend_conf']['cmvn_file']}")

    print("----------config----------")
    print(params["config"])

    os.system(f"cat {params['config']}")


def main():
    model, params = load_model("iic/SenseVoiceSmall")
    # model, params = load_model("../SenseVoiceSmall")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    display_params(params)

    generate_tokens(params, OUTPUT_DIR)

    model.__class__.forward = modified_forward
    model.eval()

    x = torch.randn(1, 100, 560, dtype=torch.float32)
    language = torch.tensor([0], dtype=torch.int32)
    text_norm = torch.tensor([15], dtype=torch.int32)

    opset_version = 13
    filename = os.path.join(OUTPUT_DIR, "sensevoice_small.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (x, language, text_norm),
            filename,
            opset_version=opset_version,
            input_names=["x", "language", "text_norm"],
            output_names=["logits"],
            dynamic_axes={
                "x": {0: "N", 1: "T"},
                "language": {0: "N"},
                "text_norm": {0: "N"},
                "logits": {0: "N", 1: "T"},
            },
            dynamo=False,
        )

    lfr_window_size = params["frontend_conf"]["lfr_m"]
    lfr_window_shift = params["frontend_conf"]["lfr_n"]

    neg_mean, inv_stddev = load_cmvn(params["frontend_conf"]["cmvn_file"])
    vocab_size = params["tokenizer"].sp.vocab_size()

    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "sense_voice_ctc",
        # version 1: Use QInt8
        # version 2: Use QUInt8
        "version": "2",
        "model_author": "iic",
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "comment": "iic/SenseVoiceSmall",
        "lang_auto": model.lid_dict["auto"],
        "lang_zh": model.lid_dict["zh"],
        "lang_en": model.lid_dict["en"],
        "lang_yue": model.lid_dict["yue"],  # cantonese
        "lang_ja": model.lid_dict["ja"],
        "lang_ko": model.lid_dict["ko"],
        "lang_nospeech": model.lid_dict["nospeech"],
        "with_itn": model.textnorm_dict["withitn"],
        "without_itn": model.textnorm_dict["woitn"],
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    filename_int8 = os.path.join(OUTPUT_DIR, "sensevoice_small.int8.onnx")
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        # Note that we have to use QUInt8 here.
        #
        # When QInt8 is used, C++ onnxruntime produces incorrect results
        weight_type=QuantType.QUInt8,
    )
    print(f"Export done. Output directory: {OUTPUT_DIR}")
    print(f"  - {os.path.basename(filename)}")
    print(f"  - {os.path.basename(filename_int8)}")
    print(f"  - tokens.txt")


if __name__ == "__main__":
    torch.manual_seed(20240717)
    main()
