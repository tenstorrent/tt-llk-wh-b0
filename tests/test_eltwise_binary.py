import pytest
import torch
import os
from dbd.tt_debuda_init import init_debuda
from dbd.tt_debuda_lib import write_to_device, read_words_from_device, run_elf
from pack import *
from unpack import *
import numpy as np
import random

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
    "Bfp8" : "FORMAT_BFP8",
    "Int32": "FORMAT_INT32"
}

mathop_args_dict = {
    "elwadd": "ELTWISE_BINARY_ADD",
    "elwsub": "ELTWISE_BINARY_SUB",
    "elwmul": "ELTWISE_BINARY_MUL"
}

def generate_stimuli(stimuli_format):
    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"):
        srcA = torch.rand(1024, dtype=format_dict[stimuli_format]) + 0.5
        srcB = torch.rand(1024, dtype=format_dict[stimuli_format]) + 0.5
    elif(stimuli_format == "Bfp8"):

        # pack and unpack for bfp8 is easier way of generating random stimuli
        # then extracting exponents, mantisas and normalizing 

        random_exponents = [random.randint(126, 129) for _ in range(64)]
        random_sm = [random.randint(126, 129) for _ in range(1024)]
        packed_bytes = pack_bfp8_tile(random_exponents,random_sm)
        unpacked_numbers = unpack_bfp8_tile(packed_bytes)
        res = []
        for nr in unpacked_numbers:
            res.append(nr.item())
        srcA = torch.tensor(res, dtype=torch.bfloat16)

        write_to_device("18-18",0x1b000, pack_bfp8_tile(random_exponents,random_sm))

        random_exponents = [random.randint(126, 129) for _ in range(64)]
        random_sm = [random.randint(126, 129) for _ in range(1024)]
        packed_bytes = pack_bfp8_tile(random_exponents,random_sm)
        unpacked_numbers = unpack_bfp8_tile(packed_bytes)
        res = []
        for nr in unpacked_numbers:
            res.append(nr.item())
        srcB = torch.tensor(res, dtype=torch.bfloat16)

        write_to_device("18-18",0x1c000, pack_bfp8_tile(random_exponents,random_sm))

    return srcA, srcB

def generate_golden(operation, operand1, operand2, data_format):
    if(data_format != "Bfp8"):
        tensor1_float = operand1.clone().detach().to(format_dict[data_format])
        tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    else:
        tensor1_float = operand1.clone()
        tensor2_float = operand2.clone()
    
    operations = {
        "elwadd": tensor1_float + tensor2_float,
        "elwsub": tensor1_float - tensor2_float,
        "elwmul": tensor1_float * tensor2_float
    }
    
    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()

def write_stimuli_to_l1(buffer_A, buffer_B, stimuli_format):
    if stimuli_format == "Float16_b":
        write_to_device("18-18", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_bfp16(buffer_B))    
    elif stimuli_format == "Float16":
        write_to_device("18-18", 0x1b000, pack_fp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_fp16(buffer_B))
    else: # bfp8 writing to L1 ins handled by generate_stimuli
        pass

@pytest.mark.parametrize("format", ["Bfp8"]) #,"Float16_b", "Float16"])
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", ["elwsub", "elwadd", "elwmul"])
@pytest.mark.parametrize("machine", ["wormhole"])
def test_all(format, mathop, testname, machine):
    context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(mathop, src_A, src_B, format)
    write_stimuli_to_l1(src_A, src_B, format)

    make_cmd = f"make format={format_args_dict[format]} mathop={mathop_args_dict[mathop]} testname={testname} machine={machine}"
    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "18-18", risc_id=i + 1)

    read_words_cnt = len(src_A) // (2 if format in ["Float16", "Float16_b"] else 1)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    elif( format == "Bfp8"):
        read_words_cnt = len(src_A)//4 + 64//4 # 272 for one tile

    read_data = read_words_from_device("18-18", 0x1a000, word_count=read_words_cnt)
    
    read_data_bytes = flatten_list([int_to_bytes_list(data) for data in read_data])
    
    if(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)
    elif(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif( format == "Bfp8"):
        res_from_L1 = unpack_bfp8_tile(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("make clean")

    # Mailbox checks
    assert read_words_from_device("18-18", 0x19FF4, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("18-18", 0x19FF8, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'
    assert read_words_from_device("18-18", 0x19FFC, word_count=1)[0].to_bytes(4, 'big') == b'\x00\x00\x00\x01'

    print("*"*50)
    print(golden[0:5])
    print(res_from_L1[0:5])
    print("*"*50)

    for i in range(len(golden)):
        if golden[i] != 0:
            assert np.isclose(golden[i],res_from_L1[i], rtol = 0.1, atol = 0.05)
