#include "mtl_context.h"

namespace nx::runtime::metal {
    void MTLContext::init_kernel(const std::string &name) {
        auto kernel = std::make_shared<MTLKernel>(name);
        kernel->init(m_device, m_lib);
        m_kernel_by_name[name] = kernel;
    }

    void MTLContext::init_kernels(const std::vector<std::string> &names, DtypeCategory dtype_category) {
        for (auto &name : names) {
            init_kernels(name, dtype_category);
        }
    }

    void MTLContext::init_kernels(const std::string &name, DtypeCategory dtype_category) {
        for (auto &dtype : all_dtypes) {
            if (dtype->has_category(dtype_category)) {
                init_kernel(name + "_" + dtype->get_name_str());
            }
        }
    }

    void MTLContext::init_initializer_kernels() {
        init_kernels("full", DtypeCategory::All);
        init_kernels("arange", DtypeCategory::Numeric);
        init_kernels("uniform", DtypeCategory::Float);
    }

    void MTLContext::init_unary_kernels() {
        std::vector<std::string> unary_names = {"neg", "sq"};
        std::vector<std::string> unary_float_names = {"exp", "log", "recip", "sin", "cos", "sqrt"};
        init_kernels(unary_names, DtypeCategory::Numeric);
        init_kernels(unary_float_names, DtypeCategory::Float);
    }

    void MTLContext::init_binary_kernels() {
        std::vector<std::string> binary_names = {"add", "sub", "mul", "div", "lt", "gt", "leq", "geq", "minimum", "maximum"};
        std::vector<std::string> eq_names = {"eq", "neq"};
        init_kernels(binary_names, DtypeCategory::Numeric);
        init_kernels(eq_names, DtypeCategory::All);
    }

    void MTLContext::init_reduce_kernels() {
        std::vector<std::string> reduce_names = {"sum", "max", "min", "argmax", "argmin"};
        for (auto &name : reduce_names) {
            init_kernels(name + "_all", DtypeCategory::Numeric);
            init_kernels("strided_" + name + "_all", DtypeCategory::Numeric);

            for (uint8_t i = 1; i <= 32; i <<= 1) {
                for (uint8_t j = 1; i * j <= 32; j <<= 1) {
                    init_kernels(std::format("{}_col_{}x{}", name, i, j), DtypeCategory::Numeric);
                    init_kernels(std::format("strided_{}_col_{}x{}", name, i, j), DtypeCategory::Numeric);
                }
            }
        }
    }

    void MTLContext::init_matmul_kernels() {
        init_kernels("naive_gemm2d", DtypeCategory::Numeric);
        init_kernels("tiled_gemm2d", DtypeCategory::Float);
        init_kernels("naive_gemm3d", DtypeCategory::Numeric);
        init_kernels("tiled_gemm3d", DtypeCategory::Float);
        init_kernels("strided_naive_gemm2d", DtypeCategory::Numeric);
        init_kernels("strided_tiled_gemm2d", DtypeCategory::Float);
        init_kernels("strided_naive_gemm3d", DtypeCategory::Numeric);
        init_kernels("strided_tiled_gemm3d", DtypeCategory::Float);
    }

    void MTLContext::init_copy_kernels() {
        for (auto &dtype1 : all_dtypes) {
            for (auto &dtype2 : all_dtypes) {
                init_kernel("copy_" + dtype1->get_name_str() + "_" + dtype2->get_name_str());
            }
        }
    }

    MTLContext::MTLContext(MTL::Device *mtl_device, const std::string &lib_path, MemoryProfilerPtr memory_profiler) : RuntimeContext(memory_profiler) {
        auto allocator = std::make_shared<MTLAllocator>();
        m_memory = std::make_shared<Cache>(allocator, memory_profiler);
        m_device = NS::TransferPtr<MTL::Device>(mtl_device);
        NS::String *path = NS::String::string(lib_path.c_str(), NS::ASCIIStringEncoding);
        auto url = NS::URL::fileURLWithPath(path);
        NS::Error *error = nullptr;
        m_lib = NS::TransferPtr<MTL::Library>(m_device->newLibrary(url, &error));

        if (error) {
            const std::string description = error->localizedDescription()->utf8String();
            throw std::runtime_error(description);
        }

        m_cmd_queue = NS::TransferPtr<MTL::CommandQueue>(m_device->newCommandQueue());
    }

    void MTLContext::init_kernels() {
        // Initializes kernels here
        init_initializer_kernels();
        init_unary_kernels();
        init_binary_kernels();
        init_reduce_kernels();
        init_matmul_kernels();
        init_copy_kernels();
    }

    bool MTLContext::register_kernel(const std::string &name, MTLKernelPtr kernel) {
        if (m_kernel_by_name.contains(name)) {
            return false;
        }

        m_kernel_by_name.insert(std::make_pair(name, kernel));
        return true;
    }
} // namespace nx::runtime::metal