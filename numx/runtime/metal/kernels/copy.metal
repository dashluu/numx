#include "utils.h"

template <class T, class R>
kernel void copy(
    const constant isize *offset [[buffer(0)]],
    const device T *input [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[1] + id] = input[offset[0] + id];
}

template <class T, class R>
kernel void strided_copy(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *in_stride [[buffer(3)]],
    const constant isize *out_stride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    const device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    isize in_loc = strided[0] ? get_elm_loc(id, ndim, shape, in_stride) : id;
    isize out_loc = strided[1] ? get_elm_loc(id, ndim, shape, out_stride) : id;
    output[offset[1] + out_loc] = input[offset[0] + in_loc];
}

#define def_copy(dtype, T, R)   \
template [[host_name("copy_" #dtype)]] [[kernel]] decltype(copy<T, R>) copy<T, R>;                          \
template [[host_name("strided_copy_" #dtype)]] [[kernel]] decltype(strided_copy<T, R>) strided_copy<T, R>;

def_copy(f32_f32, float, float);
def_copy(f32_i32, float, int);
def_copy(f32_i16, float, int16_t);
def_copy(f32_i8, float, int8_t);
def_copy(f32_b8, float, bool);
def_copy(i32_f32, int, float);
def_copy(i32_i32, int, int);
def_copy(i32_i16, int, int16_t);
def_copy(i32_i8, int, int8_t);
def_copy(i32_b8, int, bool);
def_copy(i16_f32, int16_t, float);
def_copy(i16_i32, int16_t, int);
def_copy(i16_i16, int16_t, int16_t);
def_copy(i16_i8, int16_t, int8_t);
def_copy(i16_b8, int16_t, bool);
def_copy(i8_f32, int8_t, float);
def_copy(i8_i32, int8_t, int);
def_copy(i8_i16, int8_t, int16_t);
def_copy(i8_i8, int8_t, int8_t);
def_copy(i8_b8, int8_t, bool);
def_copy(b8_f32, bool, float);
def_copy(b8_i32, bool, int);
def_copy(b8_i16, bool, int16_t);
def_copy(b8_i8, bool, int8_t);
def_copy(b8_b8, bool, bool);