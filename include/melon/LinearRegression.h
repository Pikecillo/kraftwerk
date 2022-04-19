#pragma once

#include <melon/LinearModel.h>
#include <melon/Random.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <iostream>
#include <iomanip>

namespace ml {
/**
   Linear model regression.
 */
template <size_t dim> class LinearRegression {
  public:
    using Input = typename LinearModel<dim>::Input;
    using Params = typename LinearModel<dim>::Params;
    using TrainingExample = std::pair<Input, double>;
    using TrainingSet = std::vector<TrainingExample>;

    void fit(const TrainingSet &trainingSet) {
        Random random;
        m_linearModel.setParams(random.uniform<dim + 1>(-0.5, 0.5));

        const double errorTolerance = 1E-3;
        double prevCost = costFunction(trainingSet);
        double absoluteError = std::numeric_limits<double>::max();
        double learningRate = 0.1;
        const int maxIter = 10000;
        int numIterations = 0;

        do {
            const auto grad = costFunctionGradient(trainingSet);
            numIterations++;

            const Params params = m_linearModel.params();
            const Params updatedParams = params - learningRate * grad;
            m_linearModel.setParams(updatedParams);

            const double currCost = costFunction(trainingSet);

            if(currCost > prevCost) {
                m_linearModel.setParams(params);
                learningRate /= 10;
                continue;
            }

            absoluteError = std::fabs(prevCost - currCost);
            prevCost = currCost;     
        } while(absoluteError > errorTolerance && numIterations < maxIter);
    }

    double predict(const Input &x) const { return m_linearModel.eval(x); }

    const LinearModel<dim> &getModel() const { return m_linearModel; }

  private:
    double costFunction(const TrainingSet& trainingSet) const {
        const double numExamples = static_cast<double>(trainingSet.size());

        double cost = 0.0;
        for(const auto& [x, y]: trainingSet) {
            double diff = m_linearModel.eval(x) - y;
            cost += (diff * diff);
        }

        return 0.5 * cost / numExamples; 
    }

    /**
     * Gradient of the cost function
     */
    Params costFunctionGradient(const TrainingSet &trainingSet) const {
        const double numExamples = static_cast<double>(trainingSet.size());

        Params grad;

        for (size_t i = 0; i < grad.size(); i++) {
            double sum = 0.0;

            for (size_t j = 0; j < trainingSet.size(); j++) {
                const auto &[x, y] = trainingSet[j];
                double diff = (m_linearModel.eval(x) - y);
                if(i < grad.size() - 1)
                    sum += diff * x[i];
                else
                    sum += diff;
            }

            grad[i] = sum / numExamples;
        }

        return grad;
    }

  private:
    LinearModel<dim> m_linearModel;
};

} // namespace ml