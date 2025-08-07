#include "profiler.h"

namespace nx::profiler {
    void enable_memory_profile() {
        Backend &backend = Backend::get_instance();
        MemoryProfilerPtr memory_profiler;

        for (const auto &[device_name, device_ctx] : backend) {
            memory_profiler = device_ctx->get_runtime_context()->get_memory_profiler();
            memory_profiler->enable();
        }
    }

    void disable_memory_profile() {
        Backend &backend = Backend::get_instance();
        MemoryProfilerPtr memory_profiler;

        for (const auto &[device_name, device_ctx] : backend) {
            memory_profiler = device_ctx->get_runtime_context()->get_memory_profiler();
            memory_profiler->disable();
        }
    }

    void save_memory_profile(const std::string &file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw UnableToOpenFileToSaveMemoryProfile(file_name);
        }

        stream_memory_profile(file);
        file.close();
    }

    void save_graph_profile(const Array &array, const std::string &file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw UnableToOpenFileToSaveGraphProfile(file_name);
        }

        stream_graph_profile(array, file);
        file.close();
    }

    void stream_memory_profile(std::ostream &stream) {
        if (!stream) {
            throw InvalidMemoryProfileStream();
        }

        Backend &backend = Backend::get_instance();
        size_t num_devices = backend.count_devices();
        size_t i = 0;
        MemoryProfilerPtr memory_profiler;
        stream << "[";

        for (const auto &[device_name, device_ctx] : backend) {
            memory_profiler = device_ctx->get_runtime_context()->get_memory_profiler();

            if (memory_profiler->is_enabled()) {
                memory_profiler->stream_profile(stream);

                if (i < num_devices - 1) {
                    stream << ",";
                }
            }
        }

        stream << "]";
    }

    void stream_graph_profile(const Array &array, std::ostream &stream) {
        if (!stream) {
            throw InvalidGraphProfileStream();
        }

        GraphProfiler graph_profiler;
        graph_profiler.stream_profile(array.get_graph(), stream);
    }
} // namespace nx::profiler