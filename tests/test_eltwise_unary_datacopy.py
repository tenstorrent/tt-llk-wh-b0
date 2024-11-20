import pytest
import torch
import os
from dbd.tt_debuda_init import init_debuda
from dbd.tt_debuda_lib import write_to_device, read_words_from_device, run_elf
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
    if(stimuli_format != "Bfp8_b"):
        #srcA = torch.rand(1024, dtype=format_dict[stimuli_format]) + 0.5
        srcA = torch.full((1024,), 3, dtype=format_dict[stimuli_format])
    else:
        size = 1024
        #srcA = torch.rand(1024, dtype=torch.bfloat16) + 0.5
        #srcA = torch.full((size,), 15.0625, dtype=torch.bfloat16)
        integer_part = torch.randint(-3, 4, (size,))  # (size,) generates a 1D tensor
        fraction = torch.randint(0, 16, (size,)) / 16.0
        srcA = integer_part.float() + fraction  # Convert to float to add fractions

    return srcA

def generate_golden(operand1,format):
    return operand1

def write_stimuli_to_l1(buffer_A, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("18-18", 0x1b000, pack_bfp16(buffer_A))
    elif stimuli_format == "Float16":
        write_to_device("18-18", 0x1b000, pack_fp16(buffer_A))
    elif stimuli_format == "Bfp8_b":
        write_to_device("18-18", 0x1b000, pack_bfp8_b(buffer_A))
    elif stimuli_format == "Int32":
        write_to_device("18-18", 0x1b000, pack_int32(buffer_A))
    elif stimuli_format == "Float32":
        write_to_device("18-18", 0x1b000, pack_fp32(buffer_A))

@pytest.mark.parametrize("format", ["Bfp8_b","Float16_b", "Float16"]) #,"Float32", "Int32"])
@pytest.mark.parametrize("testname", ["eltwise_unary_datacopy_test"])
@pytest.mark.parametrize("machine", ["wormhole"])
def test_all(format, testname, machine):
    context = init_debuda()
    src_A = generate_stimuli(format)
    golden = generate_golden(src_A,format)
    write_stimuli_to_l1(src_A, format)

    make_cmd = f"make --silent format={format_args_dict[format]} testname={testname} machine={machine}"

    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "18-18", risc_id=i + 1)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    elif( format == "Bfp8_b"):
        read_words_cnt = len(src_A)//4 + 64//4 # 272 for one tile
    elif( format == "Float32" or format == "Int32"):
        read_words_cnt = len(src_A)

    read_data = read_words_from_device("18-18", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])

    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8_b"):
        res_from_L1 = unpack_bfp8_b(read_data_bytes)
    elif( format == "Float32"):
        res_from_L1 = unpack_float32(read_data_bytes)
    elif( format == "Int32"):
        res_from_L1 = unpack_int32(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("make clean")

    # Mailbox checks
    assert read_words_from_device("18-18", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("18-18", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("18-18", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    if(format == "Float16_b" or format == "Float16" or format == "Float32"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.4
        rtol = 0.3

    print(golden[0:10])
    print(res_from_L1[0:10])

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b", "Float32", "Int32"] else torch.bfloat16)

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"
