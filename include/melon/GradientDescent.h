#pragma once

#include <cmath>

namespace ml {

template <typename FunctionType> class GradientDescent {
  public:
    using function_type = FunctionType;
    using input_type = typename function_type::input_type;

    struct HyperParameters {
        bool minimize = true;
        double learningRate = 0.1;
        double relativeErrorTolerance = 1E-3;
        int maxIter = 10000;
        double reductionFactor = 0.2;
    };

    struct Result {
        input_type optimalParameters;
        double optimalValue;
    };

    GradientDescent &withHyperParameters(const HyperParameters hyperParameters) {
        m_hyperParameters = hyperParameters;
        return *this;
    }

    Result optimize(const function_type &function, const input_type &initialParams) {
        auto relativeError = std::numeric_limits<double>::max();
        input_type params = initialParams;
        auto prevValue = function.eval(initialParams);
        int numIterations = 0;

        do {
            const auto gradient = function.gradient(params);
            const input_type updatedParams = params - m_hyperParameters.learningRate * gradient;
            const auto currValue = function.eval(updatedParams);

            if (currValue > prevValue) {
                m_hyperParameters.learningRate *= m_hyperParameters.reductionFactor;
            } else {
                params = updatedParams;
                relativeError = std::fabs(prevValue - currValue) / currValue;
                prevValue = currValue;
            }

            numIterations++;
        } while (relativeError > m_hyperParameters.relativeErrorTolerance &&
                 numIterations < m_hyperParameters.maxIter);

        return Result{params, prevValue};
    }

  private:
    HyperParameters m_hyperParameters;
};
} // namespace ml