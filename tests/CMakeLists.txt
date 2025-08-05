cmake_minimum_required(VERSION 3.30)
project(numx_test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set up GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB LIB_HEADER CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../numx/primitive/resource_list.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../numx/primitive/shape.h"
)

file(GLOB LIB_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../numx/primitive/resource_list.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../numx/primitive/shape.cpp"
)

file(GLOB TEST_HEADER CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

file(GLOB TEST_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(${PROJECT_NAME} ${TEST_SRC} ${TEST_HEADER} ${LIB_SRC} ${LIB_HEADER})
target_link_libraries(${PROJECT_NAME} GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})