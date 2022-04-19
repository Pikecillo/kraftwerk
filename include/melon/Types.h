#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
template <size_t dim> using Vector = std::array<double, dim>;
template <size_t dim> using TrainingExample = std::pair<Vector<dim>, double>;
template <size_t dim> using TrainingSet = std::vector<TrainingExample<dim>>;

template <size_t dim> double sqLength(const Vector<dim> &vec) {
    return std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
}

template <size_t dim> Vector<dim> operator*(const Vector<dim> &vec, const double s) {
    Vector<dim> result;

    std::transform(vec.begin(), vec.end(), result.begin(), [s](double x) { return x * s; });

    return result;
}

template <size_t dim> Vector<dim> operator*(const double s, const Vector<dim> &vec) {
    return vec * s;
}

template <size_t dim> Vector<dim> operator-(const Vector<dim> &lhs, const Vector<dim> &rhs) {
    Vector<dim> result;

    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](double xlhs, double xrhs) { return xlhs - xrhs; });

    return result;
}
} // namespace ml