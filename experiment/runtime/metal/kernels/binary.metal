#include "binary.h"

template <class Op, class T, class R>
kernel void binary(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *lstride [[buffer(3)]],
    const constant isize *rstride [[buffer(4)]],
    const constant isize *outstride [[buffer(5)]],
    const constant bool *strided [[buffer(6)]],
    const device T *lhs [[buffer(7)]],
    const device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    isize lloc = strided[0] ? get_elm_loc(id, ndim, shape, lstride) : id;
    isize rloc = strided[1] ? get_elm_loc(id, ndim, shape, rstride) : id;
    isize oloc = strided[2] ? get_elm_loc(id, ndim, shape, outstride) : id;
    output[offset[2] + oloc] = Op()(lhs[offset[0] + lloc], rhs[offset[1] + rloc]);
}

#define def_binary_kernels(opname, op, dtype, T, R) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, R>) binary<op, T, R>;

#define def_cmp_kernels(opname, op, dtype, T)       \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, bool>) binary<op, T, bool>;

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
