import pytest
import torch
import os
from ttlens.tt_lens_init import init_ttlens
from ttlens.tt_lens_lib import write_to_device, read_words_from_device, run_elf
from pack import *
from unpack import *

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


def generate_stimuli(stimuli_format, tile_cnt):
    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"):
        srcA = torch.rand(1024*tile_cnt, dtype=format_dict[stimuli_format]) + 0.5
        srcB = torch.rand(1024*tile_cnt, dtype=format_dict[stimuli_format]) + 0.5
    
    return srcA, srcB

def generate_golden(op, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])
    tensor2_float = operand2.clone().detach().to(format_dict[data_format])

    if(op==1):
        res = tensor1_float + tensor2_float
    elif(op==2):
        res = tensor1_float - tensor2_float
    elif(op==3):
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")
    
    return res.tolist()

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format, tile_cnt):

    buffer_B_address = 0x1a000 + 1024*tile_cnt

    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1a000, pack_bfp16(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1a000, pack_fp16(buffer_A))
        write_to_device("0,0", buffer_B_address, pack_fp16(buffer_B))

@pytest.mark.parametrize("mathop", range(1,4))
@pytest.mark.parametrize("tile_cnt", range(1,4))
@pytest.mark.parametrize("format", ["Float16_b", "Float16"])
@pytest.mark.parametrize("testname", ["multiple_tiles_eltwise_test"])
def test_multiple_kernels(format, testname,tile_cnt,mathop):

    # prepare setup for running kernels

    pack_start_address = 0x1a000 + 2*1024*tile_cnt
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(tile_cnt)]

    unpack_kernels = [2] * tile_cnt
    pack_kernels = [1] * tile_cnt
    math_kernels = [mathop] * tile_cnt

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

    #context = init_debuda()
    src_A, src_B = generate_stimuli(format,tile_cnt)
    golden = generate_golden(mathop,src_A,src_B,format)
    write_stimuli_to_l1(src_A,src_B,format,tile_cnt)

    make_cmd = f"make --silent format={format_args_dict[format]} testname={testname}"
    make_cmd += " unpack_kern_cnt="+ str(len(unpack_kernels))+ " unpack_kerns="+unpack_kerns_formatted
    make_cmd += " math_kern_cnt="+ str(len(math_kernels))+ " math_kerns="+math_kerns_formatted
    make_cmd += " pack_kern_cnt="+ str(len(pack_kernels))+ " pack_kerns="+pack_kerns_formatted
    make_cmd += " pack_addr_cnt="+ str(len(pack_addresses))+ " pack_addrs="+pack_addresses_formatted
    make_cmd += " unpack_a_addr_cnt="+str(tile_cnt)

    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

    os.system("make clean")

    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    #check resluts from multiple tiles

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
    read_data = read_words_from_device("0,0", pack_start_address, word_count=read_words_cnt*tile_cnt)
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)

    if(format == "Float16" or format == "Float16_b"):
        chunk_size = 512
    else:
        chunk_size = 1024
    
    res_sublists = [res_from_L1[i:i + chunk_size] for i in range(0, len(res_from_L1), chunk_size)]

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    for sublist in res_sublists:
        for i in range(len(sublist)):  
            assert torch.isclose(torch.tensor(res_from_L1[i]),torch.tensor(golden[i]), rtol = rtol, atol = atol)