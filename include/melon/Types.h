#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
template <size_t dim> using Vector = std::array<double, dim>;
template <size_t dim> using TrainingExample = std::pair<Vector<dim>, double>;
template <size_t dim> using TrainingSet = std::vector<TrainingExample<dim>>;

template <size_t dim> Vector<dim> operator+(const Vector<dim> &vec, const double s) {
    Vector<dim> result;
    std::transform(vec.begin(), vec.end(), result.begin(), [s](double x) { return x + s; });
    return result;
}

template <size_t dim> void operator+=(Vector<dim> &lhs, const Vector<dim> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                   [](double xlhs, double xrhs) { return xlhs + xrhs; });
}

template <size_t dim> Vector<dim> operator-(const Vector<dim> &lhs, const Vector<dim> &rhs) {
    Vector<dim> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](double xlhs, double xrhs) { return xlhs - xrhs; });
    return result;
}

template <size_t dim> Vector<dim> operator*(const Vector<dim> &lhs, const Vector<dim> &rhs) {
    Vector<dim> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](double xlhs, double xrhs) { return xlhs * xrhs; });
    return result;
}

template <size_t dim> Vector<dim> operator*(const Vector<dim> &vec, const double s) {
    Vector<dim> result;
    std::transform(vec.begin(), vec.end(), result.begin(), [s](double x) { return x * s; });
    return result;
}

template <size_t dim> Vector<dim> operator*(const double s, const Vector<dim> &vec) {
    return vec * s;
}

template <size_t dim> void operator/=(Vector<dim> &vec, const double s) {
    std::transform(vec.begin(), vec.end(), vec.begin(), [s](double x) { return x / s; });
}

template <size_t dim> Vector<dim> operator/(const Vector<dim> &lhs, const Vector<dim> &rhs) {
    Vector<dim> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                   [](double xlhs, double xrhs) { return xlhs / xrhs; });
    return result;
}

template <size_t dim> Vector<dim> operator/(const Vector<dim> &vec, const double s) {
    return vec * (1.0 / s);
}

template <typename T> double sqLength(const T &vec) {
    return std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
}

template <> double sqLength(const double &sc) { return sc * sc; }

template <size_t dim>
Vector<dim> apply(const Vector<dim> &vec, std::function<double(double)> &&func) {
    Vector<dim> result;
    std::transform(vec.begin(), vec.end(), result.begin(), func);
    return result;
}
} // namespace ml