#include <melon/GradientDescent.h>

#include <gtest/gtest.h>

const double tolerance = 0.1;

TEST(TestGradientDescent, optimize) {
    class QuadraticFunction {
      public:
        using argument_type = double;

        double eval(double x) const { return x * x; }
        double gradient(double x) const { return 2.0 * x; }
    };

    ml::GradientDescent optimizer;
    const auto result = optimizer.optimize(QuadraticFunction(), 100.0);
    EXPECT_NEAR(result.optimalValue, 0.0, 0.001);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}