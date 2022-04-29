#pragma once

#include <melon/Types.h>

#include <numeric>

namespace ml {
template <size_t dim> class LinearModel {
  public:
    static constexpr size_t ArgumentDim = dim;
    static constexpr size_t NumParameters = ArgumentDim + 1;

    using argument_type = Vector<ArgumentDim>;
    using parameters_type = Vector<NumParameters>;

    LinearModel() : m_parameters{} {};

    LinearModel(const parameters_type &parameters) { setParameters(parameters); }

    void setParameters(const parameters_type &parameters) { m_parameters = parameters; }

    const parameters_type &parameters() const { return m_parameters; }

    double eval(const argument_type &x) const {
        return std::inner_product(x.begin(), x.end(), m_parameters.begin(), m_parameters.back());
    }

  private:
    parameters_type m_parameters;
};
} // namespace ml