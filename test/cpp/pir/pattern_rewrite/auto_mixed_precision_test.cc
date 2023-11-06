// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/auto_mixed_precision_pass.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/pir/transforms/dead_code_elimination_pass.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32}, 1.0);

  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());

  paddle::dialect::FullOp full_d_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::FullOp full_d_out_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{32, 32}, 1.5);

  paddle::dialect::AddGradOp add_grad_op1 =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op1.out(), full_bias_op1.out(), full_d_out_op1.out());

  paddle::dialect::MatmulGradOp matmul_grad_op1 =
      builder.Build<paddle::dialect::MatmulGradOp>(
          full_input_op1.out(), full_weight_op1.out(), add_grad_op1.x_grad());

  paddle::dialect::Add_Op add__op1 = builder.Build<paddle::dialect::Add_Op>(
      full_d_weight_op1.out(), matmul_grad_op1.y_grad());

  builder.Build<paddle::dialect::FetchOp>(add_op1.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(add_grad_op1.y_grad(), "dbias", 1);
  builder.Build<paddle::dialect::FetchOp>(add__op1.out(), "dweight", 2);
}

TEST(AutoMixedPrecisonTest, MixedPrecisionTest) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 13u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateAutoMixedPrecisionPass());
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  // pm.EnablePassTiming();
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
}
