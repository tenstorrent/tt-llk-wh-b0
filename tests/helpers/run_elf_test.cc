// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "ckernel_addr_map.h"
#include "ckernel_pcbuf.h"
#include "ckernel_main.h"
#include "ckernel_globals.h"
#include <l1_address_map.h>
#include <tensix.h>
// Necessary for ckernel variables
#include "ckernel_helper.h"

#ifdef LLK_TRISC_UNPACK
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FFC);
	#ifdef MULTIPLE_OPS
		#include "operations/unpack_kernels.h"
	#endif
#elif defined(LLK_TRISC_MATH)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF8);
	#ifdef MULTIPLE_OPS
		#include "operations/math_kernels.h"
	#endif
#elif defined(LLK_TRISC_PACK)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF4);
	#ifdef MULTIPLE_OPS
		#include "operations/pack_kernels.h"
	#endif
#endif

int main()
{
    FWEVENT("Launching proudction env kernels");
	for (int i = 0; i < 64; i++) regfile[i] = 0;
	reset_cfg_state_id();
	reset_dest_offset_id();

	#ifdef LLK_TRISC_MATH
	TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0);
	#endif

	#ifdef MULTIPLE_OPS
	// needs these 2 defines when compiling
	PROCESS_NUMBERS(KERN_CNT, KERNS);
		#ifdef LLK_TRISC_PACK
			PROCESS_ADDRESSES(PACK_ADDR_CNT,PACK_ADDRS);
		#endif
	#endif

	tensix_sync();
    run_kernel();

	*mailbox = KERNEL_COMPLETE; // 0x1

	for(;;){}
}