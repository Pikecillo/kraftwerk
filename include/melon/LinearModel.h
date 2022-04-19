#pragma once

#include <melon/Types.h>

#include <numeric>

namespace ml {
template <size_t dim> class LinearModel {
  public:
    static constexpr size_t NumVars = dim;
    static constexpr size_t NumParams = NumVars + 1;
    using Input = Vector<NumVars>;
    using Params = Vector<NumParams>;

    LinearModel() = default;

    LinearModel(const Params &params) { setParams(params); }

    void setParams(const Params &params) { m_params = params; }

    const Params &params() const { return m_params; }

    double eval(const Input &x) const {
        return std::inner_product(x.begin(), x.end(), m_params.begin(),
                                  m_params.back());
    }

  private:
    Params m_params{};
};
} // namespace ml