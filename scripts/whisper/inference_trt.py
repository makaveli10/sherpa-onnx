#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import argparse

import numpy as np
import tensorrt as trt
import torch
import torchaudio
from cuda import cudart

import common

import kaldi_native_fbank as knf


class TensorRTInfer:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
        print("sizes")
        print(len(self.outputs))
        print(len(self.allocations))

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return [(self.outputs[i]["shape"], self.outputs[i]["dtype"]) for i in range(len(self.outputs))]

    def infer(self, batch, top=1):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        # Prepare the output data
        output_specs = self.output_spec()
        output1 = np.zeros(*output_specs[0])
        output2 = np.zeros(*output_specs[1])

        # Process I/O and execute the network
        common.memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        common.memcpy_device_to_host(output1, self.outputs[0]["allocation"])
        common.memcpy_device_to_host(output2, self.outputs[1]["allocation"])

        # Process the results
        print(output1.shape, output2.shape)

        return output1, output2


def compute_features(filename: str) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = torchaudio.load(filename)
    audio = wave[0].contiguous()  # only use the first channel
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=16000
        )

    features = []
    online_whisper_fbank = knf.OnlineWhisperFbank(knf.FrameExtractionOptions())
    online_whisper_fbank.accept_waveform(16000, audio.numpy())
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    target = 3000
    mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]), "constant", 0)
    mel = mel.t().unsqueeze(0)

    return mel.numpy()


def main(args):
    trt_infer = TensorRTInfer(args.engine)
    mel = compute_features(args.sound_file)
    print(mel.shape)
    output = trt_infer.infer(mel)
    print(output[0][0,0,0,:5], output[1][0,0,0,:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="The TensorRT engine to infer with")
    parser.add_argument(
        "sound_file",
        type=str,
        help="Path to the test wave",
    )
    args = parser.parse_args()
    main(args)