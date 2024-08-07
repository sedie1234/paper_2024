/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NNPAAccelerator.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator support for the IBM Telum coprocessor.
//
//===---------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

class PAAccelerator final : public Accelerator {
private:
  static PAAccelerator *instance;
  PAAccelerator();

public:
  
  PAAccelerator(PAAccelerator &) = delete;
  void operator=(const PAAccelerator &) = delete;

  ~PAAccelerator();

  static PAAccelerator *getInstance();

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const Accelerator *accel) {
    return accel->getKind() == Accelerator::Kind::PA;
  }
  static bool classof(const PAAccelerator *) { return true; }

  uint64_t getVersionNumber() const final;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void registerPasses(int optLevel) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-to-krnl pass
  //===--------------------------------------------------------------------===//
  virtual mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const final;
  virtual void conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const final;
  // virtual int64_t getDefaultAllocAlignment(
  //     const mlir::TensorType tensorType) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for krnl-to-llvm pass
  //===--------------------------------------------------------------------===//
  virtual void conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const final;
  virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) const final;
};

} // namespace accel
} // namespace onnx_mlir
