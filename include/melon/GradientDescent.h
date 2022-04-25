#pragma once

#include <cmath>

namespace ml {

template <typename InputType> class Function {
    public:
    using input_type = InputType;
    using output_type = InputType;
    virtual double eval(const InputType &input) const = 0;
    virtual InputType gradient(const InputType &input) const = 0;
};

template <typename FunctionType> class GradientDescent {
  public:
    struct HyperParameters {
        bool minimize = true;
        double learningRate = 0.1;
        double relativeErrorTolerance = 1E-3;
        int maxIter = 10000;
        double reductionFactor = 0.2;
    };

    struct Result {
        typename FunctionType::input_type optimalParameters;
        typename FunctionType::output_type optimalValue;
    };

    using input_type = typename FunctionType::input_type;
    using output_type = typename FunctionType::output_type;

    GradientDescent& withHyperParameters(const HyperParameters hyperParameters) {
        m_hyperParameters = hyperParameters;
        return *this;
    }

    Result optimize(const FunctionType& function, input_type initialParams) {
        double relativeError = std::numeric_limits<double>::max();
        input_type params = initialParams;
        double prevValue = function.eval(initialParams);
        int numIterations = 0;

        do {
            const auto gradient = function.gradient(params);
            const input_type updatedParams = params - m_hyperParameters.learningRate * gradient;
            const double currValue = function.eval(updatedParams);

            if (currValue > prevValue) {
                m_hyperParameters.learningRate *= m_hyperParameters.reductionFactor;
            }
            else {
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
} // namespace melon