#ifndef PARAMS_H
#define PARAMS_H

#ifdef LLK_TRISC_UNPACK

    #ifdef FORMAT_FLOAT16_B
        #define DATA_FORMAT (uint32_t)DataFormat::Float16_b
    #endif
    #ifdef FORMAT_FLOAT16
        #define DATA_FORMAT (uint32_t)DataFormat::Float16
    #endif
    #ifdef FORMAT_FLOAT32
        #define DATA_FORMAT (uint32_t)DataFormat::Float32
    #endif
    #ifdef FORMAT_INT32
        #define DATA_FORMAT (uint32_t)DataFormat::Int32
    #endif
    #ifdef FORMAT_BFP8
        #define DATA_FORMAT (uint32_t)DataFormat::Bfp8 
    #endif

#endif

#ifdef LLK_TRISC_MATH

    #ifdef FORMAT_FLOAT16_B
        #define DATA_FORMAT (uint32_t)DataFormat::Float16_b
    #endif
    #ifdef FORMAT_FLOAT16
        #define DATA_FORMAT (uint32_t)DataFormat::Float16
    #endif
    #ifdef FORMAT_FLOAT32
        #define DATA_FORMAT (uint32_t)DataFormat::Float32
    #endif
    #ifdef FORMAT_INT32
        #define DATA_FORMAT (uint32_t)DataFormat::Int32
    #endif
    #ifdef FORMAT_BFP8
        #define DATA_FORMAT (uint32_t)DataFormat::Bfp8 
    #endif

    #ifdef ELTWISE_BINARY_ADD
        #define ELTWISE_BINARY_OP EltwiseBinaryType::ELWADD
    #endif
    #ifdef ELTWISE_BINARY_SUB
        #define ELTWISE_BINARY_OP EltwiseBinaryType::ELWSUB
    #endif
    #ifdef ELTWISE_BINARY_MUL
        #define ELTWISE_BINARY_OP EltwiseBinaryType::ELWMUL
    #endif

    #ifdef SFPU_OP_SQRT
        #define SFPU_OPERATION sqrt
        #define SFPU_CALLS _init_sqrt_<false>();_calculate_sqrt_<false,0,10>(10);
    #endif
    #ifdef SFPU_OP_LOG
        #define SFPU_OPERATION log
        #define SFPU_CALLS _init_log_<false>();_calculate_log_<false,false,10>(10,0);
    #endif
    #ifdef SFPU_OP_SQUARE
        #define SFPU_OPERATION square
        #define SFPU_CALLS _calculate_square_<false,10>(10);
    #endif

#endif

#ifdef LLK_TRISC_PACK

    #ifdef FORMAT_FLOAT16_B
        #define DATA_FORMAT (uint32_t)DataFormat::Float16_b
    #endif
    #ifdef FORMAT_FLOAT16
        #define DATA_FORMAT (uint32_t)DataFormat::Float16
    #endif
    #ifdef FORMAT_FLOAT32
        #define DATA_FORMAT (uint32_t)DataFormat::Float32
    #endif
    #ifdef FORMAT_INT32
        #define DATA_FORMAT (uint32_t)DataFormat::Int32
    #endif
    #ifdef FORMAT_BFP8
        #define DATA_FORMAT (uint32_t)DataFormat::Bfp8 
    #endif

#endif

#endif