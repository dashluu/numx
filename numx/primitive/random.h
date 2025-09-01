#pragma once

#include "../utils.h"

namespace nx::primitive {
    using namespace nx::utils;

    struct RandomKeyGenerator : public std::enable_shared_from_this<RandomKeyGenerator> {
    private:
        uint64_t m_key;
        uint64_t m_counter = 1;

        void update_counter();

    public:
        explicit RandomKeyGenerator(uint64_t seed) : m_key(seed) {}
        RandomKeyGenerator(const RandomKeyGenerator &) = delete;
        RandomKeyGenerator(RandomKeyGenerator &&) noexcept = delete;
        ~RandomKeyGenerator() = default;
        RandomKeyGenerator &operator=(const RandomKeyGenerator &) = delete;
        RandomKeyGenerator &operator=(RandomKeyGenerator &&) noexcept = delete;
        uint64_t next();
    };

    using RandomKeyGeneratorPtr = std::shared_ptr<RandomKeyGenerator>;
    inline static std::optional<uint64_t> s_seed;
    uint64_t get_current_time_seed();

    inline uint64_t seed() {
        if (!s_seed) {
            s_seed = get_current_time_seed();
        }

        return *s_seed;
    }

    inline static constexpr uint s_rot2x32[] = {13, 15, 26, 6, 17, 29, 16, 24};
    inline uint rotl32(uint x, uint N) { return (x << (N & 31)) | (x >> ((32 - N) & 31)); }
    uint64_t threefry2x32(uint64_t key, uint64_t counter);
} // namespace nx::primitive