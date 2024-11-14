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
    exponent = exponent - 127
    for mantissa in bfp8_mantissas:
        print("mantissa dec: ", mantissa)
        sign_mantissa = str(format(mantissa, '08b'))
        sign = sign_mantissa[0]
        mantissa_value = sign_mantissa[1:]
        print("UNPACKER: ", sign_mantissa)
        mantissa_int = int(mantissa_value[:exponent+1],2) # +1 is because according to standard . is after 1 digit, not at the beginning
        mantissa_frac_bin  = mantissa_value[exponent:]
        mantissa_frac = 0
        for i in range(0,len(mantissa_frac_bin)):
            if(mantissa_frac_bin[i] == '1'):
                mantissa_frac += 1/(2**i)

        #print(mantissa_int, mantissa_frac)
        bfloat16_values.append(((-1)**int(sign,2))*(mantissa_int+mantissa_frac))

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