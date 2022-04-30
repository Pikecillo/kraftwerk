#include <melon/GradientDescent.h>

#include <gtest/gtest.h>

const double tolerance = 0.1;

TEST(TestGradientDescent, optimize) {
    class ParaboloidFunction {
      public:
        using argument_type = ml::Vector<2>;
        using gradient_type = argument_type;

        double eval(const argument_type &x) const {
            return (x[0] - 20.0) * (x[0] - 20.0) + x[1] * x[1] + 50.0;
        }
        gradient_type gradient(const argument_type &x) const {
            return {2.0 * (x[0] - 20.0), 2 * x[1]};
        }
    };

    ml::GradientDescent optimizer;
    const auto result = optimizer.optimize(ParaboloidFunction(), {100.0, 100.0});
    EXPECT_NEAR(result.optimalValue, 50.0, 0.001);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}