// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, bool SHIFT_RIGHT, int ITERATIONS>
inline void _calculate_shift_(const uint dst_offset) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(0,4,3,0);
        TT_SFPLOAD(1,4,3,dst_offset*dst_tile_size);
        // if shift amount is negative -> result should be 0
        TTI_SFPSETCC(0,1,0,0);
        TTI_SFPMOV(0,9,0,0);
        TTI_SFPENCC(0,0,0,0);
        // if shift amount is >32 -> result should be 0
        TTI_SFPIADD(0xFE0,1,2,5); // 0xFE0 = -32
        TTI_SFPSETCC(0,2,0,4);
        TTI_SFPMOV(0,9,0,0);
        TTI_SFPENCC(0,0,0,0);
        if constexpr (SHIFT_RIGHT) {
            // take negative of LREG1 to shift right
            TTI_SFPIADD(0,9,1,6);
        }
        TTI_SFPSHFT(0,1,0,0);
        TTI_SFPSTORE(0,4,3,0);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
