#pragma once

#include <melon/GradientDescent.h>
#include <melon/LinearModel.h>
#include <melon/Random.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace ml {

template <typename TModel> class CostFunction {
  public:
    using model_type = TModel;
    using argument_type = typename model_type::parameters_type;
    using gradient_type = argument_type;
    using training_set_type = TrainingSet<model_type::ArgumentDim>;

    CostFunction(const training_set_type &trainingSet) : m_trainingSet(trainingSet) {}

    gradient_type gradient(const argument_type &input) const {
        const double numExamples = static_cast<double>(m_trainingSet.size());
        const model_type model(input);
        gradient_type grad;

        for (size_t i = 0; i < grad.size(); i++) {
            double sum = 0.0;

            for (size_t j = 0; j < m_trainingSet.size(); j++) {
                const auto &[x, y] = m_trainingSet[j];
                const double diff = (model.eval(x) - y);
                if (i < grad.size() - 1)
                    sum += diff * x[i];
                else
                    sum += diff;
            }

            grad[i] = sum;
        }

        grad /= numExamples;

        return grad;
    }

  protected:
    const training_set_type &m_trainingSet;
};

/**
   Linear model regression.
 */
template <typename TModel, typename TCostFunction> class Regression {
  public:
    static constexpr size_t ArgumentDim = TModel::ArgumentDim;
    static constexpr size_t NumParameters = TModel::NumParameters;

    using model_type = TModel;
    using cost_function_type = TCostFunction;
    using argument_type = typename model_type::argument_type;
    using parameters_type = typename model_type::parameters_type;
    using training_set_type = TrainingSet<ArgumentDim>;

    static_assert(std::is_same<typename cost_function_type::argument_type, parameters_type>::value);

    virtual ~Regression() = default;

    void fit(const training_set_type &trainingSet) {
        const training_set_type &adjustedTrainingSet = adjustTrainingSet(trainingSet);
        GradientDescent gradientDescent;
        const auto &costFunction = getCostFunction(adjustedTrainingSet);
        const auto initialParameters = Random().uniform<parameters_type>(-0.5, 0.5);
        const auto result = gradientDescent.optimize(costFunction, initialParameters);

        m_model.setParameters(result.optimalArguments);
    }

    virtual double predict(const argument_type &x) const { return m_model.eval(adjustInput(x)); }

  protected:
    virtual cost_function_type getCostFunction(const training_set_type &trainingSet) = 0;

    /**
     * Adjust argument value to account for feature scaling and mean normalization.
     */
    argument_type adjustInput(const argument_type &x) const { return (x - m_means) / m_sdevs; }

    argument_type computePerFeatureMean(const training_set_type &trainingSet) {
        argument_type means = {0.0};

        for (const auto &example : trainingSet) {
            const auto &x = example.first;
            means += x;
        }

        const auto numExamples = static_cast<double>(trainingSet.size());
        means /= numExamples;

        return means;
    }

    argument_type computePerFeatureSDev(const training_set_type &trainingSet,
                                        const argument_type &means) {
        argument_type sdevs = {0};
        for (const auto &example : trainingSet) {
            const auto &x = example.first;
            const auto diff = x - means;
            sdevs += (diff * diff);
        }

        const auto numExamples = static_cast<double>(trainingSet.size());
        sdevs = apply<ArgumentDim>(sdevs / numExamples, [](double x) { return sqrt(x); });

        return sdevs;
    }

    /**
     * Perform feature scaling and mean normalization on trainingSet.
     */
    training_set_type adjustTrainingSet(const training_set_type &trainingSet) {
        m_means = computePerFeatureMean(trainingSet);
        m_sdevs = computePerFeatureSDev(trainingSet, m_means);

        training_set_type adjustedTrainingSet;

        for (const auto &[x, y] : trainingSet) {
            adjustedTrainingSet.emplace_back(adjustInput(x), y);
        }

        return adjustedTrainingSet;
    }

  protected:
    argument_type m_means, m_sdevs;
    model_type m_model;
};
} // namespace ml