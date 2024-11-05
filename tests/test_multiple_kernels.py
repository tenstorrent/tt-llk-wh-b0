import pytest
import torch
import os
import struct
from dbd.tt_debuda_init import init_debuda
from dbd.tt_debuda_lib import write_to_device, read_words_from_device, run_elf
from pack import *
from unpack import *
import random
import itertools


format_dict = {
    "Float32": torch.float32,
    "Float16": torch.float16,
    "Float16_b": torch.bfloat16,
    "Int32": torch.int32
}

format_args_dict = {
    "Float32": "FORMAT_FLOAT32",
    "Float16": "FORMAT_FLOAT16",
    "Float16_b": "FORMAT_FLOAT16_B"
}

def generate_math_kernels(length):
    return list(itertools.product([1, 2], repeat=length))

def generate_stimuli(stimuli_format):
    srcA = torch.rand(1024, dtype=format_dict[stimuli_format]) + 0.5
    srcB = torch.rand(1024, dtype=format_dict[stimuli_format]) + 0.5

    #srcA = torch.full((1024,), 2, dtype=format_dict[stimuli_format])
    #srcB = torch.full((1024,), 2, dtype=format_dict[stimuli_format])
    
    return srcA, srcB

def generate_golden(operations, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])
    tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    
    res = []

    for op in operations:
        if(op==1):
            res_tmp = tensor1_float + tensor2_float
        elif(op==2):
            res_tmp = tensor1_float - tensor2_float
        elif(op==3):
            res_tmp = tensor1_float * tensor2_float
        else:
            raise ValueError("Unsupported operation!")
        
        res.append(res_tmp.tolist())
    
    return res

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("18-18", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("18-18", 0x1b000, pack_fp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_fp16(buffer_B))

pack_addresses = [0x1a000,0x1d000, 0x1e000, 0x1f000, 0x2000, 0x2100, 0x2200, 0x2300, 0x2400, 0x2500]


@pytest.mark.parametrize("length", range(1,len(pack_addresses)+1))
@pytest.mark.parametrize("format", ["Float16_b"])
@pytest.mark.parametrize("testname", ["multiple_ops_test"])
@pytest.mark.parametrize("machine", ["wormhole"])
def test_multiple_kernels(format, testname, machine,length):

    unpack_kernels = [2]*length
    pack_kernels = [1]*length

    math_kernels_list = generate_math_kernels(length)
    math_kernels_list = [list(kernel) for kernel in math_kernels_list]

    print("*"*50)
    print(math_kernels_list)
    print(unpack_kernels)
    print(pack_kernels)
    print("*"*50)

    for math_kernels in math_kernels_list:

        # *********** formatting kernels

        unpack_kerns_formatted = ""
        for i in unpack_kernels:
            unpack_kerns_formatted+=str(i)+","
        unpack_kerns_formatted = unpack_kerns_formatted[:-1]

        math_kerns_formatted = ""
        for i in math_kernels:
            math_kerns_formatted+=str(i)+","
        math_kerns_formatted = math_kerns_formatted[:-1]

        pack_kerns_formatted = ""
        for i in pack_kernels:
            pack_kerns_formatted+=str(i)+","
        pack_kerns_formatted = pack_kerns_formatted[:-1]

        pack_addresses_formatted = ""
        for i in pack_addresses:
            pack_addresses_formatted+=str(hex(i)+",")
        pack_addresses_formatted = pack_addresses_formatted[:-1]

        # ******************************** 

        context = init_debuda()
        src_A, src_B = generate_stimuli(format)
        golden = generate_golden(math_kernels, src_A, src_B, format)
        write_stimuli_to_l1(src_A, src_B, format)

        make_cmd = f"make --silent format={format_args_dict[format]} testname={testname} machine={machine}"
        make_cmd += " unpack_kern_cnt="+ str(len(unpack_kernels))+ " unpack_kerns="+unpack_kerns_formatted
        make_cmd += " math_kern_cnt="+ str(len(math_kernels))+ " math_kerns="+math_kerns_formatted
        make_cmd += " pack_kern_cnt="+ str(len(pack_kernels))+ " pack_kerns="+pack_kerns_formatted
        make_cmd += " pack_addr_cnt="+ str(len(pack_addresses))+ " pack_addrs="+pack_addresses_formatted
        os.system(make_cmd)

        for i in range(3):
            run_elf(f"build/elf/{testname}_trisc{i}.elf", "18-18", risc_id=i + 1)

        os.system("make clean")

        # Mailbox checks
        assert read_words_from_device("18-18", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
        assert read_words_from_device("18-18", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
        assert read_words_from_device("18-18", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

        for index in range(len(golden)):
            read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
            read_data = read_words_from_device("18-18", pack_addresses[index], word_count=read_words_cnt)
            read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
            res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)
            curr_golden = golden[index]

            assert len(res_from_L1) == len(curr_golden)
            print("Checking all elements of golden at index: ", index)

            tolerance = 0.1

        for i in range(len(curr_golden)):
            
            if curr_golden[i] != 0:
                try:
                    assert abs((res_from_L1[i] - curr_golden[i]) / curr_golden[i]) <= tolerance, \
                        f"Failed at i = {i}, for math_kernels={math_kernels}, " \
                        f"golden={curr_golden[i]}, result={res_from_L1[i]} "
                except AssertionError:
                    print(f"Assertion failed for math_kernels={math_kernels}, index={index}, i={i}")
                    raise  # Reraise the exception to keep the test failing after printing.