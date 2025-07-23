#include "utils.h"

static constexpr constant uint s_rot2x32[] = {13, 15, 26, 6, 17, 29, 16, 24};

uint rotl32(uint x, uint N) {
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}

uint2 threefry2x32(uint2 key, uint2 counter) {
    uint ks[] = {key.x, key.y, 0x1BD11BDA ^ key.x ^ key.y};
    uint2 X = counter;
    X.x += ks[0];
    X.y += ks[1];
    short j = 1;
    
    for (short i = 0; i < 20; i++) {
        X.x += X.y;
        X.y = rotl32(X.y, s_rot2x32[i % 8]);
        X.y ^= X.x;
        
        if (i % 4 == 3) {
            X.x += ks[j % 3];
            X.y += ks[(j+1) % 3];
            X.y += j;
            j++;
        }
    }
    
    return X;
}

struct Uniform {
    template<class F, class I>
    uint64_t hash_to_float(uint64_t hash64, F low, F high) {
        F val;
        uint64_t result = 0;
        uint64_t tmp = hash64;
        uint64_t mask = 0;
        uint item_nbyte = sizeof(F);
        uint item_nbit = item_nbyte * 8;
        
        for (uint i = 0; i < item_nbyte; i++) {
            mask = (mask << 8) | 0xff;
        }
        
        for (uint i = 0; i < 8 / item_nbyte; i++) {
            val = static_cast<F>(tmp & mask) / metal::pow(F(2), F(item_nbit)) * (high - low) + low;
            result = (result << item_nbit) | *reinterpret_cast<thread I*>(&val);
            tmp >>= item_nbit;
        }
        
        return result;
    }
};

template<class F, class I>
void rand(
    const thread uint2 &key,
    const constant F &low,
    const constant F &high,
    const constant isize &numel,
    device uint64_t *output,
    uint id)
{
    // Compute Threefry hash
	uint ctr = id * 2;
    uint2 hash = threefry2x32(key, uint2(ctr, ctr + 1));
    uint64_t hash64 = (static_cast<uint64_t>(hash.x) << 32) | static_cast<uint64_t>(hash.y);
    
    // Transform hash into random float for each block of size TSize
    uint64_t result = Uniform().template hash_to_float<F, I>(hash64, low, high);
    
    // Masking to ensure the overflowed portion of the output is unaffected
    isize size = numel * sizeof(F);
    
    if (id < size / 8) {
        output[id] = result;
        return;
    }
    
    isize mask_size = size % 8;
    uint64_t mask = 0;
        
    for (isize i = 0; i < mask_size; i++) {
        mask = (mask << 8) | 0xff;
    }
    
    output[id] = (output[id] & ~mask) | (result & mask);
}

template<class F, class I>
kernel void uniform(
    const constant isize &key [[buffer(0)]],
    const constant F &low [[buffer(1)]],
    const constant F &high [[buffer(2)]],
    const constant isize &numel [[buffer(3)]],
    device uint64_t *output [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    rand<F, I>(uint2((key >> 32) & 0xffffffff, key & 0xffffffff), low, high, numel, output, id);
}

#define def_uniform() \
template [[host_name("uniform_f32")]] [[kernel]] decltype(uniform<float, uint32_t>) uniform<float, uint32_t>;    \
template [[host_name("uniform_f16")]] [[kernel]] decltype(uniform<half, uint16_t>) uniform<half, uint16_t>;

def_uniform()