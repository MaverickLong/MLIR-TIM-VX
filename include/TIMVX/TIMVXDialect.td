//===- TIMVXDialect.td - TIMVX dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TIMVX_DIALECT
#define TIMVX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TIM-VX dialect definition.
//===----------------------------------------------------------------------===//

def TIMVX_Dialect : Dialect {
    let name = "timvx";
    let summary = "A dialect with mapping to VeriSilicon TIM-VX";
    let description = [{
        A dialect with mapping to VeriSilicon TIM-VX
    }];
    let cppNamespace = "::mlir::timvx";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base TIM-VX operation definition.
//===----------------------------------------------------------------------===//

class TIMVX_Op<string mnemonic, list<Trait> traits = []> :
        Op<TIMVX_Dialect, mnemonic, traits>;

#endif // TIMVX_DIALECT
