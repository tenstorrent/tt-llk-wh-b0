# pack.py

import struct
import torch

def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]

def int_to_bytes_list(n):
    binary_str = bin(n)[2:].zfill(32)
    return [int(binary_str[i:i + 8], 2) for i in range(0, 32, 8)]

def float16_to_bytes(value):
    float16_value = torch.tensor(value, dtype=torch.float16)
    packed_bytes = struct.pack('>e', float16_value.item())
    return list(packed_bytes) + [0] * (4 - len(packed_bytes))

def bfloat16_to_bytes(number):
    number_unpacked = struct.unpack('!I', struct.pack('!f', number))[0]
    res_masked = number_unpacked & 0xFFFF0000
    return int_to_bytes_list(res_masked)

def pack_bfp16(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = bfloat16_to_bytes(torch_tensor[i])
        half2 = bfloat16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2][::-1], half2[0:2][::-1]][::-1])  # reverse endian
    return flatten_list(packed_bytes)

def pack_fp16(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = float16_to_bytes(torch_tensor[i])
        half2 = float16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2][::-1], half2[0:2][::-1]][::-1])  # reverse endian
    return flatten_list(packed_bytes)
