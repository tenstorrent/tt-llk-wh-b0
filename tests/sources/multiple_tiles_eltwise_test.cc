#include <cstdint>
#include <cstdio>

#include "llk_defs.h"
#include "ckernel.h"
#include "../helpers/params.h"

// Globals
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
volatile uint32_t tt_l1_ptr l1_buffer[16] __attribute__ ((section (".text#"))) __attribute__ ((aligned (16)));

#ifdef LLK_TRISC_UNPACK
#include "../helpers/operations/unpack_kernels.h"
#elif defined(LLK_TRISC_MATH)
#include "../helpers/operations/math_kernels.h"
#elif defined(LLK_TRISC_PACK)
#include "../helpers/operations/pack_kernels.h"
#endif

#if defined(LLK_TRISC_PACK)  || defined(LLK_TRISC_UNPACK)
void(*kernels[10])(int);
#else
void(*kernels[10])(void);
#endif

void run_kernel(){
    for(int i = 0; i < KERN_CNT; i++) {
        if (kernels[i]) {
            #if defined(LLK_TRISC_PACK) || defined(LLK_TRISC_UNPACK)
                kernels[i](i);
            #else
                kernels[i]();
            #endif     
        } else{
            // WRITE SOMETHING DIFFERENT FROM 1 TO MAILBOX
        }
    }
}