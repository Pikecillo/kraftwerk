enable_testing()

include_directories(${MELON_HOME}/include)

find_package(OpenSSL REQUIRED)

add_executable(TestLinearModel
    TestLinearModel.cpp
)
target_link_libraries(TestLinearModel
    gtest
    gtest_main
    pthread
)

add_executable(TestRandom
    TestRandom.cpp
)
target_link_libraries(TestRandom
    gtest
    gtest_main
    pthread
)

add_executable(TestLinearRegression
    TestLinearRegression.cpp
)
target_link_libraries(TestLinearRegression
    gtest
    gtest_main
    pthread
)

add_executable(TestTypes
    TestTypes.cpp
)
target_link_libraries(TestTypes
    gtest
    gtest_main
    pthread
)