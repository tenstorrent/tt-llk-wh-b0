#include <cstdint>
#include <cstdio>

#include "llk_defs.h"
#include "ckernel.h"

const bool unpack_to_dest = true;

// Globals
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
volatile uint32_t tt_l1_ptr l1_buffer[16] __attribute__ ((section (".text#"))) __attribute__ ((aligned (16)));

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "../helpers/params.h"

volatile uint32_t* buffer_A = (volatile uint32_t*)0x1b000;

void run_kernel()
{
    _llk_unpack_A_hw_configure_<>(DATA_FORMAT,DATA_FORMAT,FACE_R_DIM,0,4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, DATA_FORMAT, DATA_FORMAT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>((((uint32_t)buffer_A)/16)-1, 0, DATA_FORMAT, DATA_FORMAT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_eltwise_unary_datacopy.h"
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "../helpers/params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel()
{
    // copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, false, false>(0, 0, 4, DATA_FORMAT);
    _llk_math_pack_sync_init_<DstSync::SyncFull,false>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT, DATA_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, false, unpack_to_dest>(0, DATA_FORMAT, DATA_FORMAT);
    
    // calculation of sfpu operation on dest
    _llk_math_eltwise_unary_sfpu_init_<SFPU_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncFull>(0);
    // calling sfpu function from ckernel
    // this part is where parametrization of operation takes part
    #ifdef SFPU_CALLS
    SFPU_CALLS
    #endif

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncFull,false>();
}

#endif 

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "../helpers/params.h"

volatile uint32_t* buffer_Dest = (volatile uint32_t*)0x1a000;
void run_kernel()
{
    for(int i = 0; i < 16*16*4; i++)
    {
        buffer_Dest[i] = 0xdeadbeef;
    }
    _llk_pack_hw_configure_(DATA_FORMAT, DATA_FORMAT, 16*16*4);
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(DATA_FORMAT);
    #ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncFull,DstTileFaceLayout::RowMajor,false>();
    #else
    _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, false, false>();
    #endif
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull>(0, (std::uint32_t)buffer_Dest/16-1);
    _llk_pack_dest_section_done_<DstSync::SyncFull,false>();
}

#endif