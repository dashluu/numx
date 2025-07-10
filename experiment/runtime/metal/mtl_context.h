#pragma once

#include "../../core/dtype.h"
#include "../runner_context.h"
#include "mtl_allocator.h"
#include "mtl_kernel.h"

namespace nx::runtime::metal {
    using namespace nx::core;

    class MTLContext : public RunnerContext {
    private:
        NS::SharedPtr<MTL::Device> m_device;
        NS::SharedPtr<MTL::Library> m_lib;
        NS::SharedPtr<MTL::CommandQueue> m_cmd_queue;
        std::unordered_map<std::string, MTLKernelPtr> m_kernel_by_name;

        void init_kernel(const std::string &name);
        void init_kernels(const std::vector<std::string> &opnames, DtypeCategory dtype_category);
        void init_kernels(const std::string &opname, DtypeCategory dtype_category);
        void init_initializer_kernels();
        void init_unary_kernels();
        void init_binary_kernels();
        void init_reduce_kernels();
        void init_matmul_kernels();
        void init_copy_kernels();

    public:
        MTLContext(MTL::Device *mtl_device, const std::string &lib_path);
        bool register_kernel(const std::string &name, MTLKernelPtr kernel);
        NS::SharedPtr<MTL::Device> get_device() const { return m_device; }
        NS::SharedPtr<MTL::CommandQueue> get_cmd_queue() const { return m_cmd_queue; }
        MTLKernelPtr get_kernel(const std::string &name) const { return m_kernel_by_name.contains(name) ? m_kernel_by_name.at(name) : nullptr; }
    };

    using MTLContextPtr = std::shared_ptr<MTLContext>;
} // namespace nx::runtime::metal