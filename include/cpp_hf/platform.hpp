// platform.hpp - small portability helpers for hints/assumptions
#pragma once

#include <utility>

#if defined(__has_include)
#  if __has_include(<version>)
#    include <version>
#  endif
#endif

#ifndef HF_LIKELY
#  if defined(__clang__) || defined(__GNUC__)
#    define HF_LIKELY(x)   __builtin_expect(!!(x), 1)
#    define HF_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  else
#    define HF_LIKELY(x)   (x)
#    define HF_UNLIKELY(x) (x)
#  endif
#endif

#ifndef HF_ASSUME
#  if defined(__cpp_lib_assume) && __cpp_lib_assume >= 202207L
     using std::assume;
#    define HF_ASSUME(cond) std::assume(cond)
#  elif defined(__clang__)
#    define HF_ASSUME(cond) __builtin_assume(cond)
#  elif defined(__GNUC__)
#    define HF_ASSUME(cond) do { if(!(cond)) __builtin_unreachable(); } while(0)
#  else
#    define HF_ASSUME(cond) ((void)0)
#  endif
#endif

