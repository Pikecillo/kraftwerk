#include <melon/Random.h>

#include <gtest/gtest.h>

const double tolerance = 0.1;

TEST(TestRandom, uniform) {
    ml::Random random;
    const double lo = -7.0, hi = 7.0;

    {
        double sum = 0.0;
        const auto values = random.uniform<ml::Vector<1000>>(lo, hi);

        for (const auto &val : values) {
            EXPECT_LE(val, hi);
            EXPECT_GE(val, lo);
            sum += val;
        }

        EXPECT_NEAR(sum / values.size(), 0.0, tolerance);
    }
}

TEST(TestRandom, normal) {
    ml::Random random;

    {
        double sum = 0.0;
        const size_t numValues = 1000;
        const double mean = 4.0;
        size_t count = 0;

        while(count++ < numValues) {
            const auto val = random.normal(mean, 2.0);
            sum += val;
        }

        EXPECT_NEAR(sum / numValues, mean, tolerance);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}