#include "../numx/primitive/shape.h"
#include <gtest/gtest.h>

using namespace nx::primitive;

TEST(TestList, TestBroadcastableSameRank) {
    Shape shape({1, 2, 3, 1, 3});
    EXPECT_EQ(shape.broadcastable({2, 2, 1, 2, 1}), true);
}

TEST(TestList, TestBroadcastableDiffRanks) {
    Shape shape({1, 2, 3, 1, 3});
    EXPECT_EQ(shape.broadcastable({1, 2, 1}), true);
}

TEST(TestList, TestBroadcastableToSameRank) {
    Shape shape({1, 1, 1, 4, 3, 1});
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), true);
}

TEST(TestList, TestBroadcastableToDiffRanks) {
    Shape shape({1, 4, 1, 3});
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), true);
}

TEST(TestList, TestNotBroadcastableToDiffRanks) {
    Shape shape({1, 4, 1, 3, 1, 3, 3});
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), false);
}

TEST(TestList, TestNotBroadcastable) {
    Shape shape({1, 2, 3, 1, 3});
    EXPECT_EQ(shape.broadcastable({1, 2, 1, 1}), false);
}

TEST(TestList, TestBroadcastSameRank) {
    Shape shape({1, 2, 3, 1, 3});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 2, 1, 2, 1});
    ShapeView broadcast_view{2, 2, 3, 2, 3};
    ShapeStride broadcast_stride{0, 9, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestBroadcastDiffRanksV1) {
    Shape shape({1, 2, 3, 1, 3});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({1, 2, 1});
    ShapeView broadcast_view{1, 2, 3, 2, 3};
    ShapeStride broadcast_stride{18, 9, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestBroadcastDiffRanksV2) {
    Shape shape({1, 4, 1, 3});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view{2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride{0, 0, 0, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestBroadcastScalarToDims) {
    Shape shape({1});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 3, 4});
    ShapeView broadcast_view{2, 3, 4};
    ShapeStride broadcast_stride{0, 0, 0};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestBroadcastToSameRank) {
    Shape shape({1, 1, 1, 4, 3, 1});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast_to({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view{2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride{0, 12, 0, 3, 1, 0};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestBroadcastToDiffRanks) {
    Shape shape({1, 4, 1, 3});
    auto [broadcast_shape, broadcast_dims] = shape.broadcast_to({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view{2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride{0, 0, 0, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestList, TestPermuteV1) {
    Shape shape({2, 3, 4});
    Shape permute_shape = shape.permute({2, 0, 1});
    ShapeView permute_view{4, 2, 3};
    ShapeStride permute_stride{1, 12, 4};
    EXPECT_EQ(permute_shape.get_view(), permute_view);
    EXPECT_EQ(permute_shape.get_stride(), permute_stride);
}

TEST(TestList, TestPermuteV2) {
    Shape shape({2, 3, 4, 5});
    Shape permute_shape = shape.permute({3, 1, 2, 0});
    ShapeView permute_view{5, 3, 4, 2};
    ShapeStride permute_stride{1, 20, 5, 60};
    EXPECT_EQ(permute_shape.get_view(), permute_view);
    EXPECT_EQ(permute_shape.get_stride(), permute_stride);
}