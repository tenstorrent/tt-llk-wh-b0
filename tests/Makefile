TOOL_PATH?=sfpi/compiler/bin
BUILD_DIR ?=build

# Tool paths
GXX=$(TOOL_PATH)/riscv32-unknown-elf-g++
OBJDUMP=$(TOOL_PATH)/riscv32-unknown-elf-objdump
OBJCOPY=$(TOOL_PATH)/riscv32-unknown-elf-objcopy
READELF=$(TOOL_PATH)/riscv32-unknown-elf-readelf

# GCC options
OPTIONS_ALL=-O3 -march=rv32im_xttwh -mtune=rvtt-b1 -mabi=ilp32 -std=c++17 -g -ffast-math -flto
OPTIONS_COMPILE=-fno-use-cxa-atexit -fno-exceptions -Wall -Werror -Wno-unknown-pragmas -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -DARCH_WORMHOLE -DTENSIX_FIRMWARE -DDEBUG_PRINT_ENABLED
OPTIONS_LINK=-fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -Wl,--trace
INCLUDES = -I../llk_lib -I../common/inc -I../common/inc/sfpu
INCLUDES += -I../firmware/riscv/common -I../firmware/riscv/wormhole -I../firmware/riscv/wormhole/wormhole_b0_defines -I../firmware/riscv/wormhole/noc -Isfpi/include

FORMAT_ARG:=-D$(format) 

ifeq ($(mathop),)
    MATHOP_ARG =
else
	MATHOP_ARG:=-D$(mathop) 
endif

# If test needs to run muliple consecutive operations it needs some additional
# files to be linked to it
ifeq ($(testname), multiple_ops_test)
	MULTIPLE_OPS:=-DMULTIPLE_OPS
	OPTIONS_UNPACK:=-DKERN_CNT=$(unpack_kern_cnt) 
	OPTIONS_UNPACK+=-DKERNS=$(unpack_kerns) 
	OPTIONS_MATH:=-DKERN_CNT=$(math_kern_cnt) 
	OPTIONS_MATH+=-DKERNS=$(math_kerns) 
	OPTIONS_PACK:=-DKERN_CNT=$(pack_kern_cnt) 
	OPTIONS_PACK+=-DKERNS=$(pack_kerns) 
	OPTIONS_PACK+=-DPACK_ADDR_CNT=$(pack_addr_cnt) 
	OPTIONS_PACK+=-DPACK_ADDRS=$(pack_addrs) 
    LINK_OBJECTS = $(BUILD_DIR)/unpack.o $(BUILD_DIR)/math.o $(BUILD_DIR)/pack.o
else ifeq ($(testname), multiple_tiles_eltwise_test)
	MULTIPLE_OPS:=-DMULTIPLE_OPS
	OPTIONS_UNPACK:=-DKERN_CNT=$(unpack_kern_cnt) 
	OPTIONS_UNPACK+=-DKERNS=$(unpack_kerns) 
	OPTIONS_UNPACK+=-DUNPACK_A_ADDR_CNT=$(unpack_a_addr_cnt)
	OPTIONS_UNPACK+=-DUNPACK_B_ADDR_CNT=$(unpack_b_addr_cnt)
	OPTIONS_UNPACK+=-DUNPACK_A_ADDRS=$(unpack_a_addrs)
	OPTIONS_UNPACK+=-DUNPACK_B_ADDRS=$(unpack_b_addrs)
	OPTIONS_UNPACK+=-DTILE_SIZE_ELEMENTS=1024
	OPTIONS_MATH:=-DKERN_CNT=$(math_kern_cnt) 
	OPTIONS_MATH+=-DKERNS=$(math_kerns) 
	OPTIONS_PACK:=-DKERN_CNT=$(pack_kern_cnt) 
	OPTIONS_PACK+=-DKERNS=$(pack_kerns) 
	OPTIONS_PACK+=-DPACK_ADDR_CNT=$(pack_addr_cnt) 
	OPTIONS_PACK+=-DPACK_ADDRS=$(pack_addrs) 
    LINK_OBJECTS = $(BUILD_DIR)/unpack.o $(BUILD_DIR)/math.o $(BUILD_DIR)/pack.o
else	
	MULTIPLE_OPS= 
	LINK_OBJECTS= 
	OPTIONS_UNPACK= 
	OPTIONS_MATH= 
	OPTIONS_PACK= 
endif

OPTIONS_COMPILE+=$(FORMAT_ARG) $(MATHOP_ARG)
OPTIONS_COMPILE+=$(INCLUDES)

# Define project paths
RISCV_SOURCE=helpers
OUTPUT_ELFS = $(BUILD_DIR)/elf

# Define targets
.PHONY: all clean

all: $(BUILD_DIR) $(OUTPUT_ELFS) $(OUTPUT_ELFS)/$(testname)_trisc0.elf $(OUTPUT_ELFS)/$(testname)_trisc1.elf $(OUTPUT_ELFS)/$(testname)_trisc2.elf
dis: $(BUILD_DIR)/$(testname)_trisc0.dis $(BUILD_DIR)/$(testname)_trisc1.dis $(BUILD_DIR)/$(testname)_trisc2.dis

$(BUILD_DIR)/%.dis :  $(OUTPUT_ELFS)/%.elf
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

$(BUILD_DIR):
	mkdir -p $@

$(OUTPUT_ELFS): $(BUILD_DIR)
	mkdir -p $@

# Buiding .elf files for every TRISC core
 $(OUTPUT_ELFS)/$(testname)_trisc0.elf :$(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_unpack.o $(BUILD_DIR)/$(testname)_unpack.o $(LINK_OBJECTS)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(RISCV_SOURCE)/memory.wormhole.ld -T$(RISCV_SOURCE)/trisc0.ld -T$(RISCV_SOURCE)/sections.ld -o $@
 $(OUTPUT_ELFS)/$(testname)_trisc1.elf: $(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_math.o $(BUILD_DIR)/$(testname)_math.o $(LINK_OBJECTS)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(RISCV_SOURCE)/memory.wormhole.ld -T$(RISCV_SOURCE)/trisc1.ld -T$(RISCV_SOURCE)/sections.ld -o $@
 $(OUTPUT_ELFS)/$(testname)_trisc2.elf: $(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_pack.o $(BUILD_DIR)/$(testname)_pack.o $(LINK_OBJECTS)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(RISCV_SOURCE)/memory.wormhole.ld -T$(RISCV_SOURCE)/trisc2.ld -T$(RISCV_SOURCE)/sections.ld -o $@

#compiling _test.cc to .o
$(BUILD_DIR)/$(testname)_unpack.o: sources/$(testname).cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_UNPACK) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/$(testname)_math.o: sources/$(testname).cc 
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_MATH) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/$(testname)_pack.o: sources/$(testname).cc 
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_PACK) -DLLK_TRISC_PACK -c -o $@ $<

#compiling main for every TRISC core
$(BUILD_DIR)/main_unpack.o : helpers/run_elf_test.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_UNPACK) $(MULTIPLE_OPS) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/main_math.o : helpers/run_elf_test.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_MATH) $(MULTIPLE_OPS) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/main_pack.o : helpers/run_elf_test.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_PACK) $(MULTIPLE_OPS) -DLLK_TRISC_PACK -c -o $@ $<	

#compiling .cc for all operation libraries
$(BUILD_DIR)/unpack.o: helpers/operations/unpack_kernels.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_UNPACK) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/math.o: helpers/operations/math_kernels.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_MATH) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/pack.o: helpers/operations/pack_kernels.cc
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_PACK) -DLLK_TRISC_PACK -c -o $@ $<

$(BUILD_DIR)/tmu-crt0.o: $(RISCV_SOURCE)/tmu-crt0.S | $(BUILD_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)