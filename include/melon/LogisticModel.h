#pragma once

#include <melon/LinearModel.h>

#include <cmath>

namespace ml {
template <size_t dim> class LogisticModel {
  public:
    static constexpr size_t ArgumentDim = dim;
    static constexpr size_t NumParams = ArgumentDim + 1;

    using argument_type = Vector<ArgumentDim>;
    using parameters_type = Vector<NumParams>;

    LogisticModel() = default;

    LogisticModel(const parameters_type &parameters) { setParameters(parameters); }

    void setParameters(const parameters_type &parameters) {
        m_linearModel.setParameters(parameters);
    }

    const parameters_type &parameters() const { return m_linearModel.parameters(); }

    double eval(const argument_type &x) const {
        return 1.0 / (1.0 + std::exp(-m_linearModel.eval(x)));
    }

  private:
    LinearModel<ArgumentDim> m_linearModel;
};
} // namespace ml
