#include <melon/LogisticRegression.h>
#include <melon/Random.h>

#include <gtest/gtest.h>

namespace {

template <size_t dim>
ml::TrainingSet<dim> createSyntheticTrainingSet(const ml::LogisticModel<dim> &model,
                                                const size_t numExamples) {
    ml::TrainingSet<dim> trainingSet;
    size_t count = 0;
    ml::Random random;

    while (count++ < numExamples) {
        const auto x = random.uniform<ml::Vector<dim>>(-10.0, 10.0);
        trainingSet.emplace_back(x, model.eval(x));
    }

    return trainingSet;
}
} // namespace

TEST(TestLogisticRegression, predict) {
    ml::LogisticModel<10> model({3.0, 1.0, -4.0, 10.0, 1.5, -1.5, 3.0, 4.7 - 4.7, -10.0, 4.5});
    const size_t numExamples = 1000;
    const auto trainingSet = createSyntheticTrainingSet(model, numExamples);

    ml::LogisticRegression<10> regression;
    regression.fit(trainingSet);

    ml::Random random;
    const size_t numTests = 1000;
    const double errorTolerance = 0.02;
    double avgError = 0.0;
    for (size_t i = 0; i < numTests; i++) {
        const auto x = random.uniform<ml::Vector<10>>(-10.0, 10.0);
        avgError += std::fabs(model.eval(x) - regression.predict(x));
    }

    avgError /= static_cast<double>(numTests);

    EXPECT_LT(avgError, errorTolerance);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}