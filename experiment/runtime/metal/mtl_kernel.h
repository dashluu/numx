#pragma once

#include "../../utils.h"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace nx::runtime {
    class MTLKernel : public std::enable_shared_from_this<MTLKernel> {
    private:
        // shared ptr gets released once kernel is released
        NS::SharedPtr<MTL::Function> m_function;
        NS::SharedPtr<MTL::ComputePipelineState> m_state;
        std::string m_name;

    public:
        MTLKernel(const std::string &name) : m_name(name) {}

        void init(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib) {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            auto ns_name = NS::String::string(m_name.c_str(), NS::UTF8StringEncoding);
            m_function = NS::TransferPtr<MTL::Function>(lib->newFunction(ns_name));
            // TODO: handle error
            NS::Error *error = nullptr;
            m_state = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(m_function.get(), &error));
            pool->release();
        }

        NS::SharedPtr<MTL::Function> get_function() const { return m_function; }
        NS::SharedPtr<MTL::ComputePipelineState> get_state() const { return m_state; }
    };

    using MTLKernelPtr = std::shared_ptr<MTLKernel>;
} // namespace nx::runtime