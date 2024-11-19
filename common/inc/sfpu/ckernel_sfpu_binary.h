// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

enum {
    ADD_BINARY = 0,
    SUB_BINARY = 1,
    MUL_BINARY = 2,
    DIV_BINARY = 3,
    RSUB_BINARY = 4,
    POW_BINARY = 5
};  // BINOP_MODE

sfpi_inline vFloat _calculate_sfpu_binary_power_(vFloat base, vFloat pow)
{
    ////////////////////////////
    // "normalize base to calculation range"
    ////////////////////////////
    vFloat x = setexp(base, 127);    // set exp to exp bias (put base in range of 1-2)

    // XXXXXX ask Namal? if we can derive the coefficients below to higher precision
    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D'):
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    // XXXXX try variants of the below: B'=.7122, C'=2.0869
    vFloat series_result = x * (x * (x * 0.1058f - 0.7166f) + 2.0871) + -1.4753f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    vInt exp = exexp(base);
    v_if (exp < 0) {
        exp = setsgn(~exp + 1, 1);
    }
    v_endif;

    vFloat expf = int32_to_float(exp, 0);
    vFloat vConstLn2 = 0.692871f;
    vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (base == 0.0F) { // Reload for register pressure
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    vFloat val = pow * log_result;
    ////////////////////////////
    // Take exp(pow * log_result) to produce base^pow
    ////////////////////////////
    // Force sign to 0 (make number positive)
    vFloat result = _sfpu_exp_(setsgn(val, 0));

    v_if (val < 0) {
        result = _sfpu_reciprocal_(result);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const uint dst_offset)
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        vFloat in0 = dst_reg[0];
        vFloat in1 = dst_reg[dst_offset * dst_tile_size];
        vFloat result = 0.0f;

        if constexpr (BINOP_MODE == ADD_BINARY) {
            result = in0 + in1;
        } else if constexpr (BINOP_MODE == SUB_BINARY) {
            result = in0 - in1;
        } else if constexpr (BINOP_MODE == MUL_BINARY) {
            result = in0 * in1;
        } else if constexpr (BINOP_MODE == DIV_BINARY) {
            // inversion is carried out on host side and passed down
            result = in0 * in1;
        } else if constexpr (BINOP_MODE == RSUB_BINARY) {
            result = in1 - in0;
        } else if constexpr (BINOP_MODE == POW_BINARY) {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
