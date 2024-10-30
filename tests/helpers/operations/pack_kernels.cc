#include "pack_kernels.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "../params.h"

    volatile uint32_t* buffer_Dest[3] = {(volatile uint32_t*)0x1a000,(volatile uint32_t*)0x1d000,(volatile uint32_t*)0x1e000};

    void pack_Dest_kernel(int index){    
        _llk_pack_hw_configure_(DATA_FORMAT, DATA_FORMAT, 16*16*4);
        _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(DATA_FORMAT);
        _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, false, false>();
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncFull>(0, (std::uint32_t)buffer_Dest[index]/16-1);
        _llk_pack_dest_section_done_<DstSync::SyncFull,false>();
    }
