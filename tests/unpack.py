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

def bytes_to_float32(byte_list):
    bytes_data = bytes(byte_list)
    unpacked_value = struct.unpack('>f', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.float32)

def bytes_to_int32(byte_list):
    bytes_data = bytes(byte_list)
    unpacked_value = struct.unpack('>I', bytes_data)[0]
    return torch.tensor(unpacked_value, dtype=torch.int32)

def unpack_fp16(packed_list):
    return [bytes_to_float16(packed_list[i:i + 2]).item() for i in range(0, len(packed_list), 2)]

def unpack_bfp16(packed_list):
    return [bytes_to_bfloat16(packed_list[i:i + 2]).item() for i in range(0, len(packed_list), 2)]

def unpack_float32(packed_list):
    return [bytes_to_float32(packed_list[i:i + 4]).item() for i in range(0, len(packed_list), 4)]

def unpack_int32(packed_list):
    return [bytes_to_int32(packed_list[i:i + 4]).item() for i in range(0, len(packed_list), 4)]

def bfp8_to_float_block(exponent, bfp8_mantissas):
    bfloat16_values = []
    exponent = exponent - 127
    for mantissa in bfp8_mantissas:
        sign_mantissa = str(format(mantissa, '08b'))
        sign = int(sign_mantissa[0],2)
        mantissa_value =sign_mantissa[1:]
        int_part = mantissa_value[:exponent+1]
        fract_part = mantissa_value[exponent+1:]

        if(len(int_part) != 0):
            int_value = int(int_part,2)
        else:
            int_value = 0

        fract_value = 0
        for i in range(len(fract_part)):
            if(fract_part[i] == '1'):
                fract_value += 1/(2**(i+1))

        bfloat16_values.append(((-1)**sign)*(int_value+fract_value))

    return bfloat16_values

def unpack_bfp8_b(bfp8_block):
    exponents = bfp8_block[:64]
    mantissas = bfp8_block[64:]
    
    bfloat16_values = []
    for i in range(len(exponents)):
        exponent = exponents[i]
        bfp8_mantissas = mantissas[i * 16:(i + 1) * 16]        
        reversed_chunks = []
        for j in range(0, len(bfp8_mantissas), 4):
            chunk = bfp8_mantissas[j:j+4]  # Get the next chunk of 4 elements
            reversed_chunk = chunk[::-1]  # Reverse the chunk
            reversed_chunks.extend(reversed_chunk)  # Add the reversed chunk to the list

        block_bfloat16_values = bfp8_to_float_block(exponent, reversed_chunks)
        bfloat16_values.extend(block_bfloat16_values)
    
    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)