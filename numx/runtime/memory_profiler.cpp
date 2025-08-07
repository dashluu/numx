#include "memory_profiler.h"

namespace nx::runtime {
    void MemoryProfiler::stream_memory_snapshot(std::ostream &stream, const MemorySnapshot &snapshot) {
        stream << "{";
        stream << "\"size\": " << snapshot.get_block()->get_size() << ",";
        stream << "\"start\": " << snapshot.get_alloc_time().time_since_epoch().count() << ",";
        stream << "\"end\": " << snapshot.get_free_time().time_since_epoch().count();
        stream << "}";
    }

    void MemoryProfiler::trace_alloc_block(const ArrayData &data) {
        BufferBlock *block = data.get_buffer().get_block();
        m_snapshot_by_id.emplace(data.get_id(), MemorySnapshot(block));
        m_block_memory += block->get_size();
        m_peak_memory = std::max(m_peak_memory, m_block_memory);
    }

    void MemoryProfiler::trace_free_block(const ArrayData &data) {
        MemorySnapshot &snapshot = m_snapshot_by_id.at(data.get_id());
        snapshot.stop();
        m_block_memory -= snapshot.get_block()->get_size();
    }

    void MemoryProfiler::trace_alloc_pool(isize capacity) { m_pool_memory += capacity; }
    void MemoryProfiler::trace_free_pool(isize capacity) { m_pool_memory -= capacity; }

    bool MemoryProfiler::save_profile(const std::string &file_name) {
        if (!m_enabled) {
            return false;
        }

        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw UnableToOpenFileToSaveMemoryProfile(file_name);
        }

        stream_profile(file);
        file.close();
        return true;
    }

    bool MemoryProfiler::stream_profile(std::ostream &stream) {
        if (!m_enabled) {
            return false;
        }

        if (!stream) {
            throw InvalidMemoryProfileStream();
        }

        stream << "{\"Device\": \"" << m_device->str() << "\", \"Peak memory\": " << m_peak_memory << ", \"Pool memory\": " << m_pool_memory << ", \"Block memory\": {";
        size_t num_snapshots = m_snapshot_by_id.size();
        size_t num_leaks = 0;
        size_t i = 0;

        for (const auto &[id, snapshot] : m_snapshot_by_id) {
            stream << "\"" << id << "\":";
            stream_memory_snapshot(stream, snapshot);

            if (++i < num_snapshots) {
                stream << ",";
            }

            if (snapshot.is_alive()) {
                num_leaks++;
            }
        }

        stream << "}, \"Leaks\": {";
        i = 0;

        for (const auto &[id, snapshot] : m_snapshot_by_id) {
            if (snapshot.is_alive()) {
                stream << "\"" << id << "\":";
                stream_memory_snapshot(stream, snapshot);

                if (++i < num_leaks) {
                    stream << ",";
                }
            }
        }

        stream << "}}";
        return true;
    }
} // namespace nx::runtime