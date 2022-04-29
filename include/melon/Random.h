#pragma once

#include <melon/Types.h>

#include <random>

namespace ml {
class Random {
  public:
    template <typename OutputT> OutputT uniform(const double lo, const double hi) {
        std::uniform_real_distribution<double> uniformDistribution(std::min(lo, hi),
                                                                   std::max(lo, hi));
        OutputT v;
        for (auto &elem : v) {
            elem = uniformDistribution(m_generator);
        }

        return v;
    }

    double normal(const double mean, const double stddev) {
        std::normal_distribution<double> normalDistribution(mean, stddev);
        return normalDistribution(m_generator);
    }

  private:
    std::mt19937 m_generator;
};
} // namespace ml