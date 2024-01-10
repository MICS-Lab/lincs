// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__UNREACHABLE_HPP
#define LINCS__UNREACHABLE_HPP

// Shamelessly copy-pasted from https://stackoverflow.com/a/65258501/905845
#ifdef __GNUC__
[[noreturn]] inline __attribute__((always_inline)) void unreachable() {__builtin_unreachable();}
#elif defined(_MSC_VER)
[[noreturn]] __forceinline void unreachable() {__assume(false);}
#else
inline void unreachable() {}
#endif

#endif  // LINCS__UNREACHABLE_HPP
