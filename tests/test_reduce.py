import pytest
import torch
import torch.nn.functional as F
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
    "Float16_b": "FORMAT_FLOAT16_B",
    "Bfp8_b" : "FORMAT_BFP8_B",
    "Int32": "FORMAT_INT32"
}

def generate_stimuli(stimuli_format):

    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"): 
        #srcA = torch.rand(1024, dtype = format_dict[stimuli_format]) + 0.5
        #srcB = torch.rand(1024, dtype = format_dict[stimuli_format]) + 0.5
        srcA = torch.arange(1024, dtype = format_dict[stimuli_format]) 
        srcB = torch.full((1024,1), 1,dtype = format_dict[stimuli_format])
    elif(stimuli_format == "Bfp8_b"):
        size = 1024
        integer_part = torch.randint(0, 1, (size,))  
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcA = integer_part.float() + fraction 
        integer_part = torch.randint(0, 1, (size,))  
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcB = integer_part.float() + fraction  

    return srcA, srcB

def generate_golden(operand1, operand2, data_format):
    if( data_format == "Float16" or data_format == "Float16_b"):
        tensor1_float = operand1.clone().detach().to(format_dict[data_format])
        tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    else:
        tensor1_float = operand1.clone().detach().to(format_dict["Float16_b"])
        tensor2_float = operand2.clone().detach().to(format_dict["Float16_b"])
    
    A = tensor1_float.view(32, 32) 
    column_avg = A.mean(dim=0)

    return column_avg.tolist()


def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp16(buffer_B))
    elif stimuli_format == "Bfp8_b":
        write_to_device("0,0", 0x1b000, pack_bfp8_b(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp8_b(buffer_B))



@pytest.mark.parametrize("format", ["Float16_b"])
@pytest.mark.parametrize("testname", ["reduce_test"])
def test_all(format, testname):
    #context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(src_A, src_B, format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make --silent format={format_args_dict[format]} testname={testname}"
    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    elif( format == "Bfp8_b"):
        read_words_cnt = len(src_A)//4 + 64//4 # 272 for one tile

    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])

    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8_b"):
        res_from_L1 = unpack_bfp8_b(read_data_bytes)

    assert len(res_from_L1) == len(golden) * 32

    os.system("make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    print(res_tensor[0:32])

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"
