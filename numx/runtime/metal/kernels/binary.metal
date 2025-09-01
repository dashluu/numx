#include "binary.h"

template <class Op, class T, class R>
kernel void binary(
    const constant isize *offset [[buffer(0)]],
    const device T *lhs [[buffer(1)]],
    const device T *rhs [[buffer(2)]],
    device R *output [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[2] + id] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void strided_binary(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *l_stride [[buffer(3)]],
    const constant isize *r_stride [[buffer(4)]],
    const constant isize *out_stride [[buffer(5)]],
    const constant bool *strided [[buffer(6)]],
    const device T *lhs [[buffer(7)]],
    const device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    isize l_loc = strided[0] ? get_elm_loc(id, ndim, shape, l_stride) : id;
    isize r_loc = strided[1] ? get_elm_loc(id, ndim, shape, r_stride) : id;
    isize out_loc = strided[2] ? get_elm_loc(id, ndim, shape, out_stride) : id;
    output[offset[2] + out_loc] = Op()(lhs[offset[0] + l_loc], rhs[offset[1] + r_loc]);
}

#define def_binary_kernels(opname, op, dtype, T, R) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, R>) binary<op, T, R>;                              \
template [[host_name("strided_" #opname "_" #dtype)]] [[kernel]] decltype(strided_binary<op, T, R>) strided_binary<op, T, R>;

#define def_cmp_kernels(opname, op, dtype, T)       \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, bool>) binary<op, T, bool>;                            \
template [[host_name("strided_" #opname "_" #dtype)]] [[kernel]] decltype(strided_binary<op, T, bool>) strided_binary<op, T, bool>;

#define def_binary(opname, op)                      \
def_binary_kernels(opname, op, f32, float, float);  \
def_binary_kernels(opname, op, i32, int, int);

#define def_numeric_cmp(opname, op)                 \
def_cmp_kernels(opname, op, f32, float);            \
def_cmp_kernels(opname, op, i32, int);

#define def_cmp_all(opname, op)                     \
def_numeric_cmp(opname, op);                        \
def_cmp_kernels(opname, op, b8, bool);

def_binary(add, Add);
def_binary(sub, Sub);
def_binary(mul, Mul);
def_binary(div, Div);
def_binary(minimum, Minimum);
def_binary(maximum, Maximum);
def_cmp_all(eq, Eq);
def_cmp_all(neq, Neq);
def_numeric_cmp(lt, Lt);
def_numeric_cmp(gt, Gt);
def_numeric_cmp(leq, Leq);
def_numeric_cmp(geq, Geq);
