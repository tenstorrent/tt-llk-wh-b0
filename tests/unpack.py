# unpack.py

import struct
import torch

def bytes_to_float16(byte_list):
    bytes_data = bytes(byte_list[:2])
    unpacked_value = struct.unpack('>e', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.float16)

def bytes_to_bfloat16(byte_list):
    bytes_data = bytes(byte_list[:2] + [0, 0])  # Ensure we include padding
    unpacked_value = struct.unpack('>f', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.float32)

def unpack_fp16(packed_list):
    return [bytes_to_float16(packed_list[i:i + 2]).item() for i in range(0, len(packed_list), 2)]

def unpack_bfp16(packed_list):
    return [bytes_to_bfloat16(packed_list[i:i + 2]).item() for i in range(0, len(packed_list), 2)]
