#pragma once

#include <melon/LinearModel.h>
#include <melon/Regression.h>

namespace ml {

template <size_t dim> class LinearRegressionCostFunction : public CostFunction<LinearModel<dim>> {
  public:
    using model_type = LinearModel<dim>;
    using argument_type = typename model_type::parameters_type;
    using gradient_type = argument_type;
    using training_set_type = TrainingSet<dim>;

  public:
    LinearRegressionCostFunction(const training_set_type &trainingSet)
        : CostFunction<model_type>(trainingSet) {}

    double eval(const argument_type &input) const {
        const model_type model(input);

        double cost = 0.0;
        for (const auto &[x, y] : this->m_trainingSet) {
            const double diff = model.eval(x) - y;
            cost += (diff * diff);
        }

        const double numExamples = static_cast<double>(this->m_trainingSet.size());
        return 0.5 * cost / numExamples;
    }
};

/**
   Linear model regression.
 */
template <size_t dim>
class LinearRegression : public Regression<LinearModel<dim>, LinearRegressionCostFunction<dim>> {
  public:
    using cost_function_type = LinearRegressionCostFunction<dim>;
    using training_set_type = typename cost_function_type::training_set_type;

  private:
    virtual LinearRegressionCostFunction<dim>
    getCostFunction(const training_set_type &trainingSet) {
        return LinearRegressionCostFunction<dim>(trainingSet);
    }
};

} // namespace ml