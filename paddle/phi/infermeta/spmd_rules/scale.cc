/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/scale.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"

namespace phi {
namespace distributed {
SpmdInfo ScaleInferSpmd(const DistMetaTensor& x,
                        const Scalar& scale,
                        float bias,
                        bool bias_after_scale) {
  return ElementwiseUnaryInferSpmd(x);
}
}  // namespace distributed
}  // namespace phi
