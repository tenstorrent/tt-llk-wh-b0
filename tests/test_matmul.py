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
    "Float16_b": "FORMAT_FLOAT16_B",
    "Int32": "FORMAT_INT32"
}

mathop_args_dict = {
    "elwadd": "ELTWISE_BINARY_ADD",
    "elwsub": "ELTWISE_BINARY_SUB",
    "elwmul": "ELTWISE_BINARY_MUL"
}

def generate_stimuli(stimuli_format):
    srcA = torch.rand(1024, dtype=format_dict[stimuli_format]) + 2
    srcB = torch.rand(1024, dtype=format_dict[stimuli_format]) + 2

    #srcA = torch.full((1024,), 2, dtype=format_dict[stimuli_format])
    #srcB = torch.full((1024,), 2, dtype=format_dict[stimuli_format])

    return srcA, srcB

def generate_golden(operand1, operand2, data_format):
    A_float = operand1.clone().detach().to(format_dict[data_format])
    B_float = operand2.clone().detach().to(format_dict[data_format])

    A_reshaped = A_float.view(32, 32)
    B_reshaped = B_float.view(32, 32)

    result = torch.matmul(B_reshaped, A_reshaped)

    result = result.view(-1)

    return result.tolist()

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("0,0", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("0,0", 0x1b000, pack_fp16(buffer_A))
        write_to_device("0,0", 0x1c000, pack_fp16(buffer_B))

@pytest.mark.parametrize("format", ["Float16_b", "Float16"])
@pytest.mark.parametrize("testname", ["matmul_test"])
def test_all(format, testname):

    #context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(src_A, src_B, format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make format={format_args_dict[format]} testname={testname}"
    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "0,0", risc_id=i + 1)

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)
    read_data = read_words_from_device("0,0", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    
    res_from_L1 = unpack_bfp16(read_data_bytes) if format == "Float16_b" else unpack_fp16(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("make clean")

    # Mailbox checks
    assert read_words_from_device("0,0", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("0,0", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    for i in range(len(golden)):
        if golden[i] != 0:
            assert torch.isclose(torch.tensor(golden[i]),torch.tensor(res_from_L1[i]), rtol = 0.2, atol = 0.2), f"index: {i}, golden: {golden[i]}, from L1: {res_from_L1[i]}"
