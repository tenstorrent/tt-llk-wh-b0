#ifndef PACK_KERNELS_HPP
#define PACK_KERNELS_HPP

    #include <cstdint>
    #include <cstdarg>
    #include <cstdint>
    #include <cstdlib>
    #include <cstring>
    #include <string>
    #include <array>
    #include <sstream>
    #include <vector>
    #define PROCESS_NUMBERS(n, ...) processNumbers(n, __VA_ARGS__)
    #define PROCESS_ADDRESSES(n, ...) process_addresses(n, __VA_ARGS__)

    void pack_Dest_kernel(int);
    inline void nop(int){};
    extern void(*kernels[10])(int);
    extern volatile uint32_t* buffer_Dest[PACK_ADDR_CNT];

    inline void process_addresses(int n, int first, ...) {

        buffer_Dest[0] = (volatile uint32_t*)first;

        va_list args;
        va_start(args, first);
        for (int i = 1; i < n; ++i) {
            int num = va_arg(args, int);
            buffer_Dest[i] = (volatile uint32_t*)num;
        }
        va_end(args);
    }

    inline void processNumbers(int n, int first, ...) {

        // Set the first kernel based on the first argument
        if(first == 1){
            kernels[0] = &pack_Dest_kernel;
        }else{
            kernels[0] = &nop;
        }

        va_list args;
        va_start(args, first);
        for (int i = 1; i < n; ++i) {
            int num = va_arg(args, int);
            if(num == 1){
                kernels[i] = &pack_Dest_kernel;
            } else {
                kernels[i] = &nop;
            }
        }
        va_end(args);
    }

#endif