// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum SfpuType {
    tanh,
    hardtanh,
    gelu,
    exponential,
    exp_with_base,
    sigmoid,
    reciprocal,
    sqrt,
    lrelu,
    power,
    square,
    tanh_derivative,
    log,
    log_with_base,
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    less_than_equal_zero,
    greater_than_zero,
    clamp,
    gelu_derivative,
    dropout,
    //abs,
    sign,
    max,
    sine,
    cosine,
    tan,
    relu_max,
    relu_min,
    cast_fp32_to_fp16a,
    sigmoid_appx,
    gelu_appx,
    elu,
    min,
    exp2,
    heaviside,
    expm1,
    signbit,
    asin,
    acos,
    atan,
    erf,
    erfc,
    rsqrt,
    isfinite,
    isinf,
    isposinf,
    isneginf,
    isnan,
    logical_not_unary,
    erfinv,
    i0,
    silu,
    mask,
    negative,
    quant_int32,
    requant_int32,
    dequant_int32,
    add_int32,
    add1,
    topk_local_sort,
    topk_merge,
    topk_rebuild,
    unary_ne,
    unary_gt,
    unary_lt,
    softplus,
    tiled_prod,
    bitwise_xor,
    bitwise_not,
    bitwise_and,
    bitwise_or,
    right_shift,
    floor,
    left_shift,
    remainder,
    fmod,
    ceil,
    unused,
    cumsum,
    fill,
    prelu,
    reshuffle_rows,
};
