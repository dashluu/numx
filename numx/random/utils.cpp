#include "utils.h"

namespace nx::random {
    uint64_t RandomKeyGenerator::next() {
        m_key = threefry2x32(m_key, 0);
        return m_key;
    }

    uint64_t get_current_time_seed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }

    uint64_t threefry2x32(uint64_t key, uint64_t counter) {
        uint key_x = key >> 32;
        uint key_y = key;
        uint ks[] = {key_x, key_y, 0x1BD11BDA ^ key_x ^ key_y};
        uint X_x = counter >> 32;
        uint X_y = counter;
        X_x += ks[0];
        X_y += ks[1];
        short j = 1;

        // 20 rounds
        for (short i = 0; i < 20; i++) {
            X_x += X_y;
            X_y = rotl32(X_y, s_rot2x32[i % 8]);
            X_y ^= X_x;

            if (i % 4 == 3) {
                X_x += ks[j % 3];
                X_y += ks[(j + 1) % 3];
                X_y += j;
                j++;
            }
        }

        return static_cast<uint64_t>(X_x) << 32 | X_y;
    }
} // namespace nx::random