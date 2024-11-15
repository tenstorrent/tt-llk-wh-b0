# pack.py

import struct
import torch
import struct

# ANSI escape codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # Reset to default color

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

def bfloat16_to_binary(value):
    float_value = value.to(torch.float32).item()
    bfloat16_bytes = bfloat16_to_bytes(float_value)
    #print(f"{format(bfloat16_bytes[0],'08b')}{format(bfloat16_bytes[1],'08b')}")
    return f"{bfloat16_bytes[0]:08b}{bfloat16_bytes[1]:08b}"

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

def float_to_bfp8_block(block):
    exponents = []
    mantissas = []
    max_exponent = -float('inf')
    
    for value in block:
        binary_str = bfloat16_to_binary(value)
        sign = binary_str[0]
        #print("number: " , binary_str)
        exponent = int(binary_str[1:9], 2)
        #print("exponent: ", exponent - 127)
        mantissa = binary_str[9:]
        mantissa = sign + "1" + mantissa[:-1]
        #print("mantissa: ", mantissa)
        #mantissa = int(mantissa,2)
        exponents.append(exponent)
        mantissas.append(mantissa)
        max_exponent = max(max_exponent, exponent)
    
    shared_exponent = max_exponent
    #print("Shared exponent: ", shared_exponent)
    mantissas_explicit = [mantissa[1:] for mantissa in mantissas]
    sign_explicit = [mantissa[0] for mantissa in mantissas]
    #print(f"{RED}{mantissas_explicit[0:10]}{RESET}")
    #print(f"{YELLOW}{sign_explicit[0:10]}{RESET}")
    mantissas_explicit = [int(mantissa,2) for mantissa in mantissas_explicit]
     
    bfp8_mantissas = []
    for i in range(len(block)):
        exponent_delta = shared_exponent - exponents[i]
        #print(f"{GREEN}{exponent_delta}{RESET}")
        shifted_mantissa = mantissas_explicit[i] >> (exponent_delta)
        bfp8_mantissas.append(int(sign_explicit[i],2) << 7 | shifted_mantissa)
        print(f"{BLUE}{format(mantissas_explicit[i],'08b')} {exponent_delta} {format(shifted_mantissa,'08b')}{RESET}")
        print(f"{YELLOW}{format(int(sign_explicit[i],2) << 7 | shifted_mantissa,'08b')}{RESET}")
    
    return shared_exponent, bfp8_mantissas

def pack_bfp8_b(tensor, block_size=16):
    flattened_tensor = tensor.flatten()
    num_blocks = len(flattened_tensor) // block_size 
    blocks = [flattened_tensor[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    
    exponents = []
    mantissas = []
    
    for block in blocks:
        shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
        exponents.append(shared_exponent)
        mantissas.extend(bfp8_mantissas)
    
    bfp8_result = exponents + mantissas
    
    return bfp8_result