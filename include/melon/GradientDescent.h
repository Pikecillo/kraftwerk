#pragma once

#include <cmath>

namespace ml {

/**
 * Gradient descent optimization.
 */
class GradientDescent {
  public:
    struct HyperParameters {
        bool minimize = true;
        double learningRate = 0.1;
        double relativeErrorTolerance = 1E-5;
        size_t maxIter = 10000;
        double reductionFactor = 0.2;
    };

    template <typename TArguments> struct Result {
        TArguments optimalArguments;
        double optimalValue;
    };

    GradientDescent &withHyperParameters(const HyperParameters hyperParameters) {
        m_hyperParameters = hyperParameters;
        return *this;
    }

    template <typename TDifferentiableFunction>
    Result<typename TDifferentiableFunction::argument_type>
    optimize(const TDifferentiableFunction &function,
             const typename TDifferentiableFunction::argument_type &initialArguments) const {
        using argument_type = typename TDifferentiableFunction::argument_type;

        argument_type arguments = initialArguments;
        auto relativeError = std::numeric_limits<double>::max();
        auto prevValue = function.eval(arguments);
        auto learningRate = m_hyperParameters.learningRate;
        size_t numIterations = 0;

        do {
            const auto gradient = function.gradient(arguments);
            const argument_type updatedArguments = arguments - learningRate * gradient;
            const auto currValue = function.eval(updatedArguments);

            if (currValue > prevValue) {
                learningRate *= m_hyperParameters.reductionFactor;
            } else {
                arguments = updatedArguments;
                relativeError = std::fabs(prevValue - currValue) / currValue;
                prevValue = currValue;
            }

            numIterations++;
        } while (relativeError > m_hyperParameters.relativeErrorTolerance &&
                 numIterations < m_hyperParameters.maxIter);

        return Result<argument_type>{arguments, prevValue};
    }

  private:
    HyperParameters m_hyperParameters;
};
} // namespace ml