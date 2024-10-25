import pytest
import torch
import os
import struct
from dbd.tt_debuda_init import init_debuda
from dbd.tt_debuda_lib import write_to_device, read_words_from_device, run_elf

from packer import pack_bfp16, pack_fp16

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


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]

def int_to_bytes_list(n):
    binary_str = bin(n)[2:]
    padded_binary = binary_str.zfill(32)
    bytes_list = [int(padded_binary[i:i + 8], 2) for i in range(0, 32, 8)]
    return bytes_list

def float16_to_bytes(value):
    float16_value = torch.tensor(value, dtype=torch.float16)
    packed_bytes = struct.pack('>e', float16_value.item())
    byte_list = list(packed_bytes)
    return byte_list + [0] * (4 - len(byte_list))

def bytes_to_float16(byte_list):
    bytes_data = bytes(byte_list[:2])
    unpacked_value = struct.unpack('>e', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.float16)

def bytes_to_bfloat16(byte_list):
    bytes_data = bytes(byte_list[:2] + [0, 0])  # Ensure we include padding
    unpacked_value = struct.unpack('>f', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.float32)

def bfloat16_to_bytes(number):
    number_unpacked = struct.unpack('!I', struct.pack('!f', number))[0]
    res_masked = number_unpacked & 0xFFFF0000
    return int_to_bytes_list(res_masked)

def unpack_fp16(packed_list):
    res = []
    for i in range(0, len(packed_list), 2):
        float_value = bytes_to_float16(packed_list[i:i+2])
        res.append(float_value.item())
    return res

def unpack_bfp16(packed_list):
    res = []
    for i in range(0, len(packed_list), 2):
        float_value = bytes_to_bfloat16(packed_list[i:i+2])
        res.append(float_value.item())
    return res

def pack_bfp16(torch_tensor):
    
    packed_bytes = []

    for i in range(0,len(torch_tensor),2):
        half1 = bfloat16_to_bytes(torch_tensor[i])
        half2 = bfloat16_to_bytes(torch_tensor[i+1])

        packed_bytes.append([half1[0:2][::-1],half2[0:2][::-1]][::-1]) # reverse endian
    
    packed_bytes = flatten_list(packed_bytes)
    packed_bytes = flatten_list(packed_bytes)

    return packed_bytes

def pack_fp16(torch_tensor):
    
    packed_bytes = []

    for i in range(0,len(torch_tensor),2):
        half1 = float16_to_bytes(torch_tensor[i])
        half2 = float16_to_bytes(torch_tensor[i+1])

        packed_bytes.append([half1[0:2][::-1],half2[0:2][::-1]][::-1]) # reverse endian
    
    packed_bytes = flatten_list(packed_bytes)
    packed_bytes = flatten_list(packed_bytes)

    return packed_bytes

def generate_stimuli(stimuli_format):
    #srcA = torch.rand(32 * 32, dtype=format_dict[stimuli_format]) + 0.5
    #srcB = torch.rand(32 * 32, dtype=format_dict[stimuli_format]) + 0.5

    srcA = torch.arange(0,512,0.5, dtype=format_dict[stimuli_format])
    srcB = torch.arange(0,512,0.5, dtype=format_dict[stimuli_format])

    return srcA, srcB

def generate_golden(operation, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format])
    tensor2_float = operand2.clone().detach().to(format_dict[data_format])
    
    if operation == "elwadd":
        dest = tensor1_float + tensor2_float
    elif operation == "elwsub":
        dest = tensor1_float - tensor2_float
    elif operation == "elwmul":
        dest = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")

    return dest.tolist()

def write_stimuli_to_l1(buffer_A, buffer_B,stimuli_format, mathop):

    if(stimuli_format == "Float16_b"):
        write_to_device("18-18", 0x1b000, pack_bfp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_bfp16(buffer_B))    
    elif(stimuli_format == "Float16"):
        write_to_device("18-18", 0x1b000, pack_fp16(buffer_A))
        write_to_device("18-18", 0x1c000, pack_fp16(buffer_B))

@pytest.mark.parametrize("format", ["Float16_b", "Float16"])
@pytest.mark.parametrize("testname", ["eltwise_binary_test"])
@pytest.mark.parametrize("mathop", ["elwadd", "elwsub", "elwmul"])
@pytest.mark.parametrize("machine", ["wormhole"])
def test_all(format, mathop, testname, machine):
    context = init_debuda()
    src_A, src_B = generate_stimuli(format)
    golden = generate_golden(mathop, src_A, src_B,format)
    write_stimuli_to_l1(src_A, src_B,format,mathop)

    make_cmd = f"make --silent format={format_args_dict[format]} mathop={mathop_args_dict[mathop]} testname={testname} machine={machine}"
    os.system(make_cmd)

    for i in range(3):
        run_elf(f"build/elf/{testname}_trisc{i}.elf", "18-18", risc_id=i + 1)

    if(format == "Float16" or format == "Float16_b"):
        read_words_cnt = len(src_A)//2
    else:
        read_words_cnt = len(src_A)

    read_data = read_words_from_device("18-18", 0x1a000, word_count=read_words_cnt)
    read_data_bytes = []

    for data in read_data:
        read_data_bytes.append(int_to_bytes_list(data))
    
    read_data_bytes = flatten_list(read_data_bytes)
    
    if(format == "Float16_b"):
        res_from_L1 = unpack_bfp16(read_data_bytes)
    elif(format == "Float16"):
        res_from_L1 = unpack_fp16(read_data_bytes)

    assert len(res_from_L1) == len(golden)

    os.system("make clean")

    unpack_mailbox = read_words_from_device("18-18", 0x19FF4, word_count=1)[0].to_bytes(4, 'big')
    math_mailbox = read_words_from_device("18-18", 0x19FF8, word_count=1)[0].to_bytes(4, 'big')
    pack_mailbox = read_words_from_device("18-18", 0x19FFC, word_count=1)[0].to_bytes(4, 'big')

    assert unpack_mailbox == b'\x00\x00\x00\x01'
    assert math_mailbox == b'\x00\x00\x00\x01'
    assert pack_mailbox == b'\x00\x00\x00\x01'

    tolerance = 0.1

    for i in range(len(golden)):
        read_bytes =  hex(read_words_from_device("18-18", 0x1a000 + i*4, word_count=1)[0])
        if(golden[i]!=0):
            assert abs((res_from_L1[i]-golden[i])/golden[i]) <= tolerance, f"i = {i}, {golden[i]}, {res_from_L1[i]} {read_bytes}"