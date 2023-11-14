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

#include "paddle/fluid/pir/transforms/auto_mixed_precision_pass.h"
#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class AutoMixedPrecisionPattern : public pir::RewritePattern {
 public:
  AutoMixedPrecisionPattern(
      pir::IrContext* context,
      const phi::Place& place,
      const phi::DataType& precision_mode,
      pir::PatternBenefit benefit = 1,
      const std::vector<std::string>& generated_names = {})
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context, generated_names) {
    precision_mode_ = precision_mode;  // should be set by user
    place_ = place;                    // should be set by user
    SetDefaultBlacklist();
    SetDefaultWhitelist();
  }

  void SetDefaultBlacklist() {
    black_list_.insert({
        paddle::dialect::ExpOp::name(),
        paddle::dialect::SquareOp::name(),
        paddle::dialect::LogOp::name(),
        paddle::dialect::FetchOp::name(),

        // paddle::dialect::Mean::name(),
        // paddle::dialect::Sum::name(),
        paddle::dialect::SigmoidCrossEntropyWithLogitsOp::name(),
    });
  }

  void SetDefaultWhitelist() {
    white_list_.insert({paddle::dialect::FullOp::name(),
                        paddle::dialect::Conv2dOp::name(),
                        paddle::dialect::TransposeOp::name()});
    return;
  }

  bool Match(pir::Operation* op) const override {
    if (op->isa<pir::GetParameterOp>() || op->isa<pir::SetParameterOp>() ||
        op->isa<paddle::dialect::CastOp>())
      return false;

    // if the op didn't support low precision, and input is in low precision,
    // insert cast op if op support low precision and input is in low precision,
    // ok otherwise, insert cast op if input is GetParameterOp, it must be
    // transformed into a low precision tensor

    return true;
  }

  std::unique_ptr<paddle::dialect::OpYamlInfoParser> GetOpYamlInfoParser(
      pir::Operation* op) const {
    paddle::dialect::OpYamlInfoInterface op_info_interface =
        op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();

    std::unique_ptr<paddle::dialect::OpYamlInfoParser> op_info_parser(nullptr);
    if (op_info_interface) {
      op_info_parser = std::make_unique<paddle::dialect::OpYamlInfoParser>(
          op_info_interface.GetOpInfo());
    }

    return op_info_parser;
  }

  std::string GetKernelFnStr(
      const paddle::dialect::OpYamlInfoParser* op_info_parser,
      pir::Operation* op_item) const {
    std::string kernel_fn_str;
    if (op_info_parser != nullptr) {
      kernel_fn_str = op_info_parser->OpRuntimeInfo().kernel_func;
    }

    if (op_item->isa<paddle::dialect::AddN_Op>() ||
        op_item->isa<paddle::dialect::AddNWithKernelOp>()) {
      if (op_item->result(0).type().isa<paddle::dialect::SelectedRowsType>()) {
        kernel_fn_str = "add_n_sr";
      }
    }
    return kernel_fn_str;
  }

  phi::Kernel GetPhiKernelSupportPrecision(pir::Operation* op,
                                           phi::Backend backend,
                                           phi::DataType precision) const {
    auto op_type = op->name();
    std::cout << "op name " << op->name() << std::endl;

    // no need for this?
    auto op_info_parser = GetOpYamlInfoParser(op);

    auto kernel_fn_str = GetKernelFnStr(op_info_parser.get(), op);
    phi::KernelKey kernel_key(backend, phi::DataLayout::ALL_LAYOUT, precision);
    auto phi_kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
        kernel_fn_str, kernel_key);

    return phi_kernel;
  }

  bool OpSupportPrecision(pir::Operation* op,
                          phi::Backend backend,
                          phi::DataType precision) const {
    auto op_type = op->name();

    // no need for this?
    auto op_info_parser = GetOpYamlInfoParser(op);

    auto kernel_fn_str = GetKernelFnStr(op_info_parser.get(), op);

    std::cout << "now check kernel " << kernel_fn_str << std::endl;

    // if the op is in white list, return true
    // if (white_list_.count(op_type)) {
    //   return true;
    // }

    // if the op is in black list, return false
    if (black_list_.count(op_type)) {
      return false;
    }

    // std::cout << KernelSupportPrecision(
    //                  "batch_norm", phi::Backend::CPU, phi::DataType::FLOAT32)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "batch_norm", phi::Backend::GPU, phi::DataType::FLOAT32)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "batch_norm", phi::Backend::CPU, phi::DataType::FLOAT16)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "batch_norm", phi::Backend::GPU, phi::DataType::FLOAT16)
    //           << std::endl;

    // std::cout << KernelSupportPrecision(
    //                  "conv2d", phi::Backend::CPU, phi::DataType::FLOAT32)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "conv2d", phi::Backend::GPU, phi::DataType::FLOAT32)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "conv2d", phi::Backend::CPU, phi::DataType::FLOAT16)
    //           << std::endl;
    // std::cout << KernelSupportPrecision(
    //                  "square", phi::Backend::GPU, phi::DataType::FLOAT16)
    //           << std::endl;
    // if the op is not in black list, and not in white list, check if the op
    // support low precision
    return KernelSupportPrecision(kernel_fn_str, backend, precision);
  }

  bool ValueInPrecision(pir::Value value, phi::DataType precision) const {
    auto dtype = pir::GetDataTypeFromValue(value);
    return paddle::dialect::TransToPhiDataType(dtype) == precision;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    phi::Backend backend = ConvertPlaceToBackend(place_);
    // if the op support low precision
    if (OpSupportPrecision(op, backend, precision_mode_)) {
      // change result's dtype to low precision
      std::cout << "change result's dtype to low precision " << op->name()
                << std::endl;

      if (op->HasAttribute("dtype")) {
        pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
            pir::IrContext::Instance(), precision_mode_);
        op->set_attribute("dtype", attr_dtype);
      }
      auto phi_kernel =
          GetPhiKernelSupportPrecision(op, backend, precision_mode_);
      auto args_def = phi_kernel.args_def();
      auto input_defs = args_def.input_defs();
      auto output_defs = args_def.output_defs();

      for (size_t i = 0; i < op->num_results(); i++) {
        auto result = op->result(i);
        paddle::dialect::DenseTensorType origin_type =
            result.type().dyn_cast<paddle::dialect::DenseTensorType>();
        auto out_phi_dtype = output_defs[i].dtype;
        pir::Type new_type = paddle::dialect::DenseTensorType::get(
            pir::IrContext::Instance(),
            paddle::dialect::TransToIrDataType(out_phi_dtype),
            origin_type.dims(),
            origin_type.data_layout(),
            origin_type.lod(),
            origin_type.offset());

        result.set_type(new_type);
      }

      // out_dtype in_dtype attribute need modify

      // fetch and feed

      // if any of the op's input is not in low precision, insert cast op
      for (size_t i = 0; i < op->num_operands(); i++) {
        auto operand = op->operand(i);
        auto in_phi_dtype = input_defs[i].dtype;
        if (!ValueInPrecision(operand.source(), in_phi_dtype)) {
          rewriter.SetInsertionPoint(op);  // before op
          paddle::dialect::CastOp cast_op =
              rewriter.Build<paddle::dialect::CastOp>(operand.source(),
                                                      in_phi_dtype);

          operand.set_source(cast_op->result(0));
        }
      }
    } else {  // current op doesn't support low precision
      // if the op's input is in low precision, insert cast op
      for (auto operand : op->operands()) {
        // get the op's dtype
        auto result_dtype = pir::GetDataTypeFromValue(op->result(0));
        if (ValueInPrecision(operand.source(), precision_mode_)) {
          rewriter.SetInsertionPoint(op);  // before op
          paddle::dialect::CastOp cast_op =
              rewriter.Build<paddle::dialect::CastOp>(
                  operand.source(),
                  paddle::dialect::TransToPhiDataType(result_dtype));

          operand.set_source(cast_op->result(0));
        }
      }
    }
  }

 private:
  std::unordered_set<std::string> black_list_;
  std::unordered_set<std::string> white_list_;
  phi::DataType precision_mode_{phi::DataType::UNDEFINED};

  phi::Place place_;

  bool PhiKernelSupportPrecision(
      const std::string& op_type,
      phi::Backend backend,
      phi::DataType data_type,
      phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) const {
    const auto& kernels = phi::KernelFactory::Instance().kernels();
    if (kernels.count(op_type) == 0) {
      return false;
    }
    phi::KernelKey kernel_key(backend, layout, data_type);
    return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
  }

  phi::Backend ConvertPlaceToBackend(const phi::Place& place) const {
    switch (place.GetType()) {
      case phi::AllocationType::CPU:
        return phi::Backend::CPU;
      case phi::AllocationType::GPU:
        return phi::Backend::GPU;
      case phi::AllocationType::XPU:
        return phi::Backend::XPU;
      default:
        return phi::Backend::UNDEFINED;
    }
    return phi::Backend::UNDEFINED;
  }

  bool KernelSupportPrecision(
      const std::string& op_type,
      phi::Backend backend,
      phi::DataType precision,
      phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) const {
    auto phi_op_type = phi::TransToPhiKernelName(op_type);

    bool support =
        PhiKernelSupportPrecision(phi_op_type, backend, precision, layout);
    if (backend == phi::Backend::GPU) {
      support |= PhiKernelSupportPrecision(
          phi_op_type, phi::Backend::GPUDNN, precision, layout);
    }
    if (!support) {
      const auto& all_kernels =
          paddle::framework::OperatorWithKernel::AllOpKernels();
      auto it = all_kernels.find(op_type);
      if (it != all_kernels.end()) {
        for (const auto& kern_pair : it->second) {
          if (ConvertPlaceToBackend(kern_pair.first.place_) == backend &&
              kern_pair.first.data_type_ ==
                  paddle::framework::TransToProtoVarType(precision)) {
            support = true;
            break;
          }
        }
      }
    }
    return support;
  }
};

class AutoMixedPrecisionPass : public pir::Pass {
 public:
  AutoMixedPrecisionPass(const phi::Place& place,
                         const phi::DataType& precision_mode)
      : pir::Pass("auto_mixed_precision_pass", 1),
        place_(place),
        precision_mode_(precision_mode) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AutoMixedPrecisionPattern>(context, place_, precision_mode_);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    // const auto& kernels = phi::KernelFactory::Instance().kernels();
    // std::cout << "all kernel size = " << kernels.size() << std::endl;
    // for (auto [k, v] : kernels) {
    //   std::cout << "kernel name " << k << std::endl;
    // }
    LOG(INFO) << "Run pass" << std::endl;
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
  phi::Place place_;
  phi::DataType precision_mode_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAutoMixedPrecisionPass(
    const phi::Place& place, const phi::DataType& precision_mode) {
  return std::make_unique<AutoMixedPrecisionPass>(place, precision_mode);
}

}  // namespace pir
