#pragma once

#include <melon/Types.h>

#include <cmath>
#include <iostream>
namespace ml {

/**
 * Gradient descent optimization.
 */
class GradientDescent {
  public:
    struct HyperParameters {
        bool minimize = true;
        double relativeErrorTolerance = 1E-5;
        size_t maxIter = 10000;
        double searchControlFactor = 0.8; // (0, 1)
        double reductionFactor = 0.8;     // (0, 1)
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
        size_t numIterations = 0;

        do {
            const auto result = backtrackingLineSearch(function, arguments);

            std::cout << "From backtacking opValue " << result.optimalValue << std::endl;

            arguments = result.optimalArguments;
            relativeError = (prevValue - result.optimalValue) / prevValue;
            prevValue = result.optimalValue;
            numIterations++;
        } while (relativeError > m_hyperParameters.relativeErrorTolerance &&
                 numIterations < m_hyperParameters.maxIter);

        return {arguments, prevValue};
    }

    template <typename TDifferentiableFunction>
    Result<typename TDifferentiableFunction::argument_type>
    backtrackingLineSearch(const TDifferentiableFunction &function,
                           const typename TDifferentiableFunction::argument_type &arguments) const {
        const auto gradient = function.gradient(arguments);
        const double localSlope = -sqLength(gradient);
        const double t = localSlope * m_hyperParameters.searchControlFactor;
        auto candidateArguments = arguments;
        auto candidateValue = function.eval(candidateArguments);
        double difference = std::numeric_limits<double>::lowest();
        double learningRate = 1.0;
        int iter = 0;

        while (difference < learningRate * t) {
            learningRate *= m_hyperParameters.reductionFactor;
            candidateArguments = arguments - learningRate * gradient;

            const double currValue = function.eval(candidateArguments);

            difference = candidateValue - currValue;
            candidateValue = currValue;
        }

        return {candidateArguments, candidateValue};
    }

  private:
    HyperParameters m_hyperParameters;
};
} // namespace ml