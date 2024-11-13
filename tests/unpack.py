# unpack.py

import struct
import torch
import struct

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

def bfp8_to_float_block(exponent, bfp8_mantissas):
    bfloat16_values = []
    
    for mantissa in bfp8_mantissas:
        sign_mantissa = str(format(mantissa, '08b'))
        sign = str(sign_mantissa[0])
        mantissa_value = sign_mantissa[1:]
        exp_bin = str(format(exponent, '08b'))
        full_number = f"{sign}{exp_bin}{mantissa_value}"
        full_number = int(full_number, 2)
        full_number_bytes = int_to_bytes_list(full_number)
        bfloat16_values.append(bytes_to_bfloat16([full_number_bytes[2], full_number_bytes[3], 0, 0]))

    return bfloat16_values

def unpack_bfp8_b(bfp8_block):
    exponents = bfp8_block[:64]
    mantissas = bfp8_block[64:]
    
    bfloat16_values = []
    for i in range(len(exponents)):
        exponent = exponents[i]
        bfp8_mantissas = mantissas[i * 16:(i + 1) * 16]
        block_bfloat16_values = bfp8_to_float_block(exponent, bfp8_mantissas)
        bfloat16_values.extend(block_bfloat16_values)
    
    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)