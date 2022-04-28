#pragma once

#include <melon/Types.h>

#include <numeric>

namespace ml {
template <size_t dim> class LinearModel {
  public:
    static constexpr size_t NumVars = dim;
    static constexpr size_t NumParams = NumVars + 1;
    using input_type = Vector<NumVars>;
    using params_type = Vector<NumParams>;

    LinearModel() = default;

    LinearModel(const params_type &params) { setParams(params); }

    void setParams(const params_type &params) { m_params = params; }

    const params_type &params() const { return m_params; }

    double eval(const input_type &x) const {
        return std::inner_product(x.begin(), x.end(), m_params.begin(), m_params.back());
    }

  private:
    params_type m_params{};
};
} // namespace ml