import torch
from .dictionaries import *

def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]

def generate_random_face(stimuli_format = "Float16_b",const_value = 1,  const_face = False):

    if(stimuli_format == "Float16" or stimuli_format == "Float16_b"): 
        #srcA_face = torch.rand(256, dtype = format_dict[stimuli_format]) + 2 # because of log
        if const_face == True:
            srcA_face = torch.ones(256, dtype = format_dict[stimuli_format]) * const_value
        else: # random for both faces
            srcA_face = torch.rand(256, dtype = format_dict[stimuli_format]) + 2 # because of log
    elif(stimuli_format == "Bfp8_b"):
        size = 256
        integer_part = torch.randint(0, 2, (size,))  
        fraction = torch.randint(0, 16, (size,)).to(dtype = torch.bfloat16) / 16.0
        srcA_face = integer_part.to(dtype = torch.bfloat16) + fraction 
    elif(stimuli_format == "Float32"):
        srcA_face = torch.arange(0,64,0.25)
    elif(stimuli_format == "Int32"):
        srcA_face = torch.arange(256)

    return srcA_face

def generate_random_face_ab(stimuli_format, const_face = False, const_value_A = 1, const_value_B = 2):
    return generate_random_face(stimuli_format,const_value_A,const_face), generate_random_face(stimuli_format,const_value_B,const_face)

def generate_stimuli(stimuli_format = "Float16_b", tile_cnt = 1, sfpu = False, const_face = False, const_value_A = 1, const_value_B = 1):

    srcA = []
    srcB = []

    for i in range(4*tile_cnt):
        face_a, face_b = generate_random_face_ab(stimuli_format, const_face,const_value_A,const_value_B)
        srcA.append(face_a.tolist())
        srcB.append(face_b.tolist())

    srcA = flatten_list(srcA)
    srcB = flatten_list(srcB)    

    if sfpu == False:
        if stimuli_format != "Bfp8_b":
            return torch.tensor(srcA, dtype = format_dict[stimuli_format]), torch.tensor(srcB, dtype = format_dict[stimuli_format])
        else:
            return torch.tensor(srcA, dtype = torch.bfloat16), torch.tensor(srcB, dtype = torch.bfloat16)
    else:
        srcA = generate_random_face(stimuli_format)
        srcB = torch.full((256,), 0)
        return srcA,srcB