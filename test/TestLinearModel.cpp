#include <melon/LinearModel.h>

#include <gtest/gtest.h>

TEST(TestLinearModel, emptyConstructor) {
    ml::LinearModel<4> model;

    EXPECT_EQ(model.eval(ml::Vector<4>{1.0, 2.0, 3.0, 4.0}), 0.0);
}

TEST(TestLinearModel, constructor) {
    ml::LinearModel<4> model({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_EQ(model.eval(ml::Vector<4>{1.0, 2.0, 3.0, 4.0}), 35.0);
}

TEST(TestLinearModel, setParameters) {
    ml::LinearModel<4> model;
    const ml::Vector<5> expected = {1.0, 2.0, 3.0, 4.0, 5.0};

    model.setParameters(expected);
    EXPECT_EQ(model.eval(ml::Vector<4>{1.0, 2.0, 3.0, 4.0}), 35.0);

    const auto current = model.parameters();
    for (size_t i = 0; i < current.size(); i++)
        EXPECT_EQ(expected[i], current[i]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}