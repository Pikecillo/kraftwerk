#pragma once

#include <melon/LinearModel.h>
#include <melon/Regression.h>

namespace ml {
template <size_t dim> class LogisticRegressionCostFunction {
  public:
    using input_type = typename RegressionTraits<dim>::params_type;
    using gradient_type = input_type;
    using training_example_type = typename RegressionTraits<dim>::training_example_type;
    using training_set_type = typename RegressionTraits<dim>::training_set_type;

  public:
    LogisticRegressionCostFunction(const training_set_type &trainingSet)
        : m_trainingSet(trainingSet) {}

    double eval(const input_type &input) const {
        LinearModel<dim> linearModel(input);
        linearModel.setParams(input);

        double cost = 0.0;
        for (const auto &[x, y] : m_trainingSet) {
            double prediction = 1.0 / 1.0 + std::exp(-linearModel.eval(x));
            cost += (y * log(prediction) + (1.0 - y) * log(1.0 - prediction));
        }

        const double numExamples = static_cast<double>(m_trainingSet.size());
        return -cost / numExamples;
    }

    gradient_type gradient(const input_type &input) const {
        const double numExamples = static_cast<double>(m_trainingSet.size());
        gradient_type grad;
        LinearModel<dim> linearModel(input);

        for (size_t i = 0; i < grad.size(); i++) {
            double sum = 0.0;

            for (size_t j = 0; j < m_trainingSet.size(); j++) {
                const auto &[x, y] = m_trainingSet[j];
                double diff = (linearModel.eval(x) - y);
                if (i < grad.size() - 1)
                    sum += diff * x[i];
                else
                    sum += diff;
            }

            grad[i] = sum / numExamples;
        }

        return grad;
    }

  private:
    const training_set_type &m_trainingSet;
};

/**
   Logistic regression.
 */
template <size_t dim>
class LogisticRegression : public Regression<LogisticRegressionCostFunction<dim>, dim> {
  public:
    using input_type = typename RegressionTraits<dim>::input_type;
    using training_set_type = typename RegressionTraits<dim>::training_set_type;

    virtual double predict(const input_type &x) const {
        const auto linearModelPrediction =
            Regression<LogisticRegressionCostFunction<dim>, dim>::predict(x);
        return 1.0 / (1.0 + std::exp(-linearModelPrediction));
    }

  private:
    virtual LogisticRegressionCostFunction<dim>
    getCostFunction(const training_set_type &trainingSet) {
        return LogisticRegressionCostFunction<dim>(trainingSet);
    }
};

} // namespace ml