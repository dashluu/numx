#include "utils.h"

template <class T, class R>
kernel void copy(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *instride [[buffer(3)]],
    const constant isize *outstride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    const device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    isize iloc = strided[0] ? get_elm_loc(id, ndim, shape, instride) : id;
    isize oloc = strided[1] ? get_elm_loc(id, ndim, shape, outstride) : id;
    output[offset[1] + oloc] = input[offset[0] + iloc];
}

#define def_copy(dtype, T, R)   \
template [[host_name("copy_" #dtype)]] [[kernel]] decltype(copy<T, R>) copy<T, R>;

def_copy(f32_f32, float, float);
def_copy(f32_i32, float, int);
def_copy(f32_b8, float, bool);
def_copy(i32_f32, int, float);
def_copy(i32_i32, int, int);
def_copy(i32_b8, int, bool);
def_copy(b8_f32, bool, float);
def_copy(b8_i32, bool, int);
def_copy(b8_b8, bool, bool);