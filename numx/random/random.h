#include "../core/array.h"

namespace nx::random {
    using namespace nx::primitive;
    using namespace nx::core;

    inline void seed(uint64_t value) { s_seed = value; }

    template <Numeric T>
    Array uniform(const ShapeView &view, T low, T high, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device_by_name(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator_by_device_name(device_name);
        uint64_t key = rand_key_gen->next();
        return Array(nx::graph::uniform(view, key, low, high, dtype, device));
    }
} // namespace nx::random