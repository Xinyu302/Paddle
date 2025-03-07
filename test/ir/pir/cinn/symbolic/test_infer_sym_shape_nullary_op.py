# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from test_infer_sym_shape_utils import (
    TestBase,
    apply_to_static,
    check_infer_results,
)

import paddle
from paddle.static import InputSpec


class ArangeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, in_0, in_1, in_2):
        if in_1 is None:
            end = in_0
            out = paddle.arange(end)
        else:
            start, end, step = in_0, in_1, in_2
            out = paddle.arange(start, end, step)

        return out


class ArangeOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.start = paddle.full([1], 0)
        self.end = paddle.full([1], 5)
        self.step = paddle.full([1], 1)
        self.expected = ['shape[Mul(Add(S1, -S0), 1 / (S2))], data[NULL]']

    def test_eval_symbolic(self):
        net = ArangeNet()
        input_spec = [
            InputSpec(shape=[None], dtype='float32'),
            InputSpec(shape=[None], dtype='float32'),
            InputSpec(shape=[None], dtype='float32'),
        ]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(
            net, input_spec, 'builtin.shadow_output', self.expected
        )
        out = net(self.start, self.end, self.step)
        return out


class EmptyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.empty(shape=[128, 32])
        out = paddle.empty(shape=x)
        return out


class EmptyOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = [
            'shape[128, 32], data[NULL]',
            'shape[S0, S1, S2], data[NULL]',
        ]

    def test_eval_symbolic(self):
        net = EmptyNet()

        x_spec = InputSpec(shape=[None, None, None], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.empty', self.expected)
        return True


class GaussianNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.tensor.random.gaussian(shape=[12, 32], mean=1.0, std=2.0)
        return out


class GaussianOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.expected = ['shape[12, 32], data[NULL]']

    def test_eval_symbolic(self):
        net = GaussianNet()
        x_spec = InputSpec(shape=[None, None, 2], dtype='float32')
        input_spec = [x_spec]
        net = apply_to_static(net, False, input_spec)
        net.eval()
        check_infer_results(net, input_spec, 'pd_op.gaussian', self.expected)
        return True


if __name__ == '__main__':
    unittest.main()
