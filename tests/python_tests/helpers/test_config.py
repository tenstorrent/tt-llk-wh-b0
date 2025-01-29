from .dictionaries import *

def generate_make_command(test_config):
    make_cmd = f"make --silent "

    input_format = test_config.get("input_format", "Float16_b") # Flolat16_b is default
    output_format = test_config.get("output_format", "Float16_b")
    testname = test_config.get("testname")
    dest_acc = test_config.get("dest_acc", " ") # default is not 32 bit dest_acc 

    make_cmd += f"format={format_args_dict[output_format]} testname={testname} dest_acc={dest_acc} " # jsut for now take output_format
    
    mathop = test_config.get("mathop", "no_mathop")

    if(mathop != "no_mathop"):
        if isinstance(mathop,str): # single tile option
            make_cmd += f"mathop={mathop_args_dict[mathop]}"
        else: # multiple tiles handles mathop as int

            if(mathop == 1):
                make_cmd += " mathop=ELTWISE_BINARY_ADD "
            elif(mathop == 2):
                make_cmd += " mathop=ELTWISE_BINARY_SUB "
            else:
                make_cmd += " mathop=ELTWISE_BINARY_MUL "

            kern_cnt = str(test_config.get("kern_cnt"))
            pack_addr_cnt = str(test_config.get("pack_addr_cnt"))
            pack_addrs = test_config.get("pack_addrs")
            unpack_a_addrs_cnt = test_config.get("unpack_a_addrs_cnt")

            make_cmd += f" kern_cnt={kern_cnt} "
            make_cmd += f" pack_addr_cnt={pack_addr_cnt} pack_addrs={pack_addrs}" 
            make_cmd += f" unpack_a_addrs_cnt={unpack_a_addrs_cnt}"



    return make_cmd