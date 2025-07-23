#include "utils.h"

template <class T>
kernel void full(
    const device T &c [[buffer(0)]],
    device T *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = c;
}

template <class T>
kernel void arange(
    const device isize &start [[buffer(0)]],
    const device isize &step [[buffer(1)]],
    device T *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = *start + static_cast<int>(id) * *step;
}

#define def_initializer_kernels(opname, op, dtype, T)   \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(op<T>) op<T>;

#define def_initializer_numeric(opname, op)             \
def_initializer_kernels(opname, op, f32, float);        \
def_initializer_kernels(opname, op, i32, int);

#define def_initializer_all(opname, op)                 \
def_initializer_numeric(opname, op);                    \
def_initializer_kernels(opname, op, b8, bool);

def_initializer_all(full, full);
def_initializer_numeric(arange, arange);