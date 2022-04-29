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
    const auto x = random.uniform<ml::Vector<10>>(-1.0, 1.0);
    ASSERT_NEAR(regression.predict(x), model.eval(x), 0.01);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}