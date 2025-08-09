#include "functional.h"

namespace nx::core {
    std::pair<isize, isize> compute_fan_in_and_fan_out(const ShapeView &view) {
        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for a view of {} dimensions, which is fewer than 2.", view.size()));
        }

        isize receptive_field_size = std::accumulate(view.begin() + 2, view.end(), 1ll, std::multiplies<>());
        isize fan_in = view[1] * receptive_field_size;
        isize fan_out = view[0] * receptive_field_size;
        // Note: fan-in and fan-out cannot be 0 since each dimension > 0
        return {fan_in, fan_out};
    }

    std::pair<isize, isize> compute_fan_in_and_fan_out(const Array &array) {
        const ShapeView &view = array.get_view();

        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for array {} of {} dimensions, which is fewer than 2.", array.get_id(), view.size()));
        }

        return compute_fan_in_and_fan_out(view);
    }
} // namespace nx::core