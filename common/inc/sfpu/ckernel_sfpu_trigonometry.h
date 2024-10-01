// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

static const float FRAC_2_PI = 0.63661975F;

template <bool APPROXIMATION_MODE>
static vFloat sfpu_sinpi_2(vFloat x);

// Approximate sin(π/2 * x) in [-0.5, 0.5] for fp32
// Maximum relative error: 3.41e-8
template <>
sfpi_inline vFloat sfpu_sinpi_2<false>(vFloat x)
{
    vFloat xx = x * x;
    return x * (((-0.00462859F
        * xx + 0.079691978F)
        * xx - 0.645965F)
        * xx + 1.5707964F);
}

// Approximate sin(π/2 * x) in [-0.5, 0.5] for fp16a
// Maximum relative error: 4.09e-4
template <>
sfpi_inline vFloat sfpu_sinpi_2<true>(vFloat x)
{
    vFloat xx = x * x;
    return (-0.6260741F * xx + 1.5701545F) * x;
}

template <bool APPROXIMATION_MODE>
static vFloat sfpu_cospi_2(vFloat x);

// Approximate cos(π/2 * x) in [-0.5, 0.5] for fp32
// Maximum relative error: 8.25e-8
template <>
sfpi_inline vFloat sfpu_cospi_2<false>(vFloat x)
{
    x *= x;
    return ((-0.020381697F
        * x + 0.25358754F)
        * x - 1.2336957F)
        * x + 0.99999994F;
}

// Approximate cos(π/2 * x) in [-0.5, 0.5] for fp16a
// Maximum relative error: 1.18e-5
template <>
sfpi_inline vFloat sfpu_cospi_2<true>(vFloat x)
{
    x *= x;
    return (0.24572742F * x - 1.2329242F) * x + 0.9999882F;
}

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
void _calculate_sine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        vFloat v = dst_reg[0];
        v *= FRAC_2_PI;

        vInt q = float_to_int16(v);
        v -= int32_to_float(q, 0);

        v_if ((q & 1) == 1) {
            v = sfpu_cospi_2<APPROXIMATION_MODE>(v);
        }
        v_else {
            v = sfpu_sinpi_2<APPROXIMATION_MODE>(v);
        }
        v_endif;

        v_if ((q & 2) == 2) {
            v = -v;
        }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void _calculate_cosine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        vFloat v = dst_reg[0];
        v *= FRAC_2_PI;

        vInt q = float_to_int16(v);
        v -= int32_to_float(q, 0);

        v_if ((q & 1) == 1) {
            v = sfpu_sinpi_2<APPROXIMATION_MODE>(v);
        }
        v_else {
            v = sfpu_cospi_2<APPROXIMATION_MODE>(v);
        }
        v_endif;

        v_if (((q + 1) & 2) == 2) {
            v = -v;
        }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
