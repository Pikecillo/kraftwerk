#include <melon/Types.h>

#include <gtest/gtest.h>

TEST(TestTypes, sqLength) {
    {
        ml::Vector<2> vec = {4.0, 3.0};
        double expected = 25.0;
        double actual = ml::sqLength(vec);
        EXPECT_EQ(expected, actual);
    }
    {
        ml::Vector<5> vec = {-4.0, 3.0, 2.0, 2.0, 2.0};
        double expected = 37.0;
        double actual = ml::sqLength(vec);
        EXPECT_EQ(expected, actual);
    }
}

TEST(TestTypes, sub) {
    {
        ml::Vector<2> lhs = {4.0, 3.0}, rhs = {-1.0, 2.0};
        ml::Vector<2> expected = {5.0, 1.0};
        auto actual = ml::operator-(lhs, rhs);
        EXPECT_EQ(expected, actual);
    }
    {
        ml::Vector<5> lhs = {14.0, 3.0, 7.0, 1.0, -8.0}, rhs = {-1.0, 2.0, -1.0, 10.0, 9.0};
        ml::Vector<5> expected = {15.0, 1.0, 8.0, -9.0, -17.0};
        auto actual = ml::operator-(lhs, rhs);
        EXPECT_EQ(expected, actual);
    }
}

TEST(TestTypes, scalarMul) {
    {
        ml::Vector<2> vec = {4.0, 3.0};
        ml::Vector<2> expected = {28.0, 21.0};
        auto actual = ml::operator*(7.0, vec);
        EXPECT_EQ(expected, actual);
    }
    {
        ml::Vector<2> vec = {4.0, -3.0};
        ml::Vector<2> expected = {28.0, -21.0};
        auto actual = ml::operator*(vec, 7.0);
        EXPECT_EQ(expected, actual);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}