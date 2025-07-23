#include "unary.h"

template <class Op, class T, class R>
kernel void unary(
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
    isize iloc = strided[0] ? get_elm_loc(id, ndim, shape, in_stride) : id;
    isize oloc = strided[1] ? get_elm_loc(id, ndim, shape, out_stride) : id;
    output[offset[1] + oloc] = Op()(input[offset[0] + iloc]);
}

#define def_unary_all_kernels(opname, op, dtype, T, R)  \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, R>) unary<op, T, R>;

#define def_unary_float_kernels(opname, op, dtype, T)   \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, float>) unary<op, T, float>;

#define def_unary_float(opname, op)                     \
def_unary_float_kernels(opname, op, f32, float);        \
def_unary_float_kernels(opname, op, i32, int);

#define def_unary_all(opname, op)                       \
def_unary_all_kernels(opname, op, f32, float, float);   \
def_unary_all_kernels(opname, op, i32, int, int);

def_unary_float(exp, Exp);
def_unary_float(log, Log);
def_unary_all(neg, Neg);
def_unary_float(recip, Recip);
def_unary_float(sin, Sin);
def_unary_float(cos, Cos);
def_unary_all(sq, Sq);
def_unary_float(sqrt, Sqrt);
