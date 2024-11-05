#include "math_kernels.h"
#include "../params.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

    void elwadd_kernel(){
    _llk_math_pack_sync_init_<DstSync::SyncFull,false>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT,DATA_FORMAT);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(4, 0, 0);
    _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE,DstSync::SyncFull>(4, 0, true);
    _llk_math_dest_section_done_<DstSync::SyncFull,false>();
    }

    void elwsub_kernel(){
    _llk_math_pack_sync_init_<DstSync::SyncFull,false>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT,DATA_FORMAT);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE>(4, 0, 0);
    _llk_math_eltwise_binary_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE,DstSync::SyncFull>(4, 0, true);
    _llk_math_dest_section_done_<DstSync::SyncFull,false>();
    }

    void elwmul_kernel(){
    _llk_math_pack_sync_init_<DstSync::SyncFull,false>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT,DATA_FORMAT);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(4, 0, 0);
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE,DstSync::SyncFull>(4, 0, true);
    _llk_math_dest_section_done_<DstSync::SyncFull,false>();
    }

    //TODO: ADD MORE