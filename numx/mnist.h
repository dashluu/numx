#include "nn/linear.h"

using namespace nx::core;
using namespace nx::nn;

class MnistModel : public Module {
private:
    std::unique_ptr<Linear> m_linear1;
    std::unique_ptr<Linear> m_linear2;

public:
    MnistModel() {
        m_linear1 = std::make_unique<Linear>(784, 128);
        m_linear2 = std::make_unique<Linear>(128, 10);
        const auto &params1 = m_linear1->get_parameters();
        const auto &params2 = m_linear2->get_parameters();
        m_params.reserve(params1.size() + params2.size());
        m_params.insert(m_params.end(), params1.begin(), params1.end());
        m_params.insert(m_params.end(), params2.begin(), params2.end());
    }

    MnistModel(const MnistModel &) = delete;
    ~MnistModel() = default;
    MnistModel &operator=(const MnistModel &) = delete;

    Array forward(const Array &x) override {
        auto x1 = (*m_linear1)(x);
        auto x2 = relu(x1);
        auto x3 = (*m_linear2)(x2);
        return x3;
    }
};