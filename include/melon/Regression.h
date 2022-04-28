#pragma once

#include <melon/GradientDescent.h>
#include <melon/LinearModel.h>
#include <melon/Random.h>

#include <utility>
#include <vector>

namespace ml {
/**
   Linear model regression.
 */
template <size_t dim> struct RegressionTraits {
    using input_type = typename LinearModel<dim>::input_type;
    using params_type = typename LinearModel<dim>::params_type;
    using training_example_type = std::pair<input_type, double>;
    using training_set_type = std::vector<training_example_type>;
};

template <typename DifferentiableFunction, size_t dim> class Regression {
  public:
    using input_type = typename RegressionTraits<dim>::input_type;
    using params_type = typename RegressionTraits<dim>::params_type;
    using training_example = typename RegressionTraits<dim>::training_example_type;
    using training_set = typename RegressionTraits<dim>::training_set_type;

    virtual ~Regression() = default;

    void fit(const training_set &trainingSet) {
        const training_set &adjustedTrainingSet = adjustTrainingSet(trainingSet);
        GradientDescent<DifferentiableFunction> gradientDescent;
        const auto &costFunction = getCostFunction(adjustedTrainingSet);
        const auto initialParams = Random().uniform<dim + 1>(-0.5, 0.5);
        const auto result = gradientDescent.optimize(costFunction, initialParams);

        m_linearModel.setParams(result.optimalParameters);
    }

    virtual double predict(const input_type &x) const { return m_linearModel.eval(adjustInput(x)); }

  protected:
    virtual DifferentiableFunction getCostFunction(const training_set &trainingSet) = 0;

    input_type adjustInput(const input_type &x) const { return (x - m_means) / m_sdevs; }

    Vector<dim> computePerFeatureMean(const training_set &trainingSet) {
        Vector<dim> means = {0.0};

        for (const auto &example : trainingSet) {
            const auto &x = example.first;
            means += x;
        }

        const auto numExamples = static_cast<double>(trainingSet.size());
        means /= numExamples;

        return means;
    }

    Vector<dim> computePerFeatureSDev(const training_set &trainingSet, const Vector<dim> &means) {
        Vector<dim> sdevs = {0};
        for (const auto &example : trainingSet) {
            const auto &x = example.first;
            const auto diff = x - means;
            sdevs += (diff * diff);
        }

        const auto numExamples = static_cast<double>(trainingSet.size());
        sdevs = apply<dim>(sdevs / numExamples, [](double x) { return sqrt(x); });

        return sdevs;
    }

    training_set adjustTrainingSet(const training_set &trainingSet) {
        m_means = computePerFeatureMean(trainingSet);
        m_sdevs = computePerFeatureSDev(trainingSet, m_means);

        training_set adjustedTrainingSet;

        for (const auto &[x, y] : trainingSet) {
            adjustedTrainingSet.emplace_back(adjustInput(x), y);
        }

        return adjustedTrainingSet;
    }

  protected:
    Vector<dim> m_means, m_sdevs;
    LinearModel<dim> m_linearModel;
};
} // namespace ml