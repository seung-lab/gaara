#ifndef _GAARA_BUILTINS_HXX_
#define _GAARA_BUILTINS_HXX_


#ifdef _MSC_VER
#  include <intrin.h>
#  define popcount __popcnt

// https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
unsigned long ctz(unsigned long value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value)) {
        return trailing_zero;
    }
    else {
        return 32;
    }
}
#else
#  define popcount __builtin_popcount
#  define ctz __builtin_ctz
#endif

#endif