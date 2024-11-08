// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
//#include "llk_sfpu_types.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"


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
    abs,
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
    reshuffle_rows,
    cumsum,
    fill
};

using namespace ckernel;
// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);

    if (sfpu_op == SfpuType::topk_local_sort) {
        addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
        }.set(ADDR_MOD_6);
    }
}
inline void eltwise_unary_sfpu_configure_mop();

template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const uint dst_index) {
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_unary_sfpu_done_() {
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_() {
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_() {
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}
