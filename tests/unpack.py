# unpack.py

import struct
import torch

def int_to_bytes_list(n):
    binary_str = bin(n)[2:].zfill(32)
    return [int(binary_str[i:i + 8], 2) for i in range(0, 32, 8)]

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

def unpack_bfp8_tile(packed_list):
    result = []
    exponents = []

    block = [[] for _ in range(64)]

    for i in range(64):
        exponents.append(packed_list[i])

    packed_list = packed_list[64:]  # Remove the first 64 bytes
    for i in range(1024):
        block[i % 16].append(packed_list[i])

    assert len(block) == 64

    # Rebuild the numbers using exponents and the sign-mantissa data
    for i in range(64):
        for j in range(16):
            sign = block[j][i] >> 7  # Extract the sign bit
            # Merge the sign, exponent, and mantissa correctly
            merged_number = (sign << 15) | (exponents[i] << 7) | (block[j][i] & 0x7F)
            merged_number_bytes = int_to_bytes_list(merged_number)[::-1]
            merged_number_bytes = [merged_number_bytes[1], merged_number_bytes[0],0,0]
            # if((i==0) and (j<5)):
            #     print(f"{YELLOW}{merged_number_bytes}{RESET}")
            result.append(bytes_to_bfloat16(merged_number_bytes))

    return result