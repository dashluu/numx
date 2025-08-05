#include "../numx/primitive/resource_list.h"
#include <gtest/gtest.h>

using namespace nx::primitive;

struct TestItem : public Resource {
private:
    int value;

public:
    TestItem(int value) : value(value) {}
    ~TestItem() = default;
    int get_value() const { return value; }
};

TestItem *push_to_list(ResourceList &list, int value) {
    TestItem *item = new TestItem(value);
    list.push(item);
    return item;
}

void populate_result(ResourceList &list, std::vector<int> &result) {
    for (Resource *resource : list) {
        TestItem *item = static_cast<TestItem *>(resource);
        result.emplace_back(item->get_value());
    }
}

void free_list(ResourceList &list) {
    for (Resource *resource : list) {
        delete static_cast<TestItem *>(resource);
    }
}

TEST(TestList, TestPush) {
    ResourceList list;
    std::vector<int> data = {3, 1, 4, 2};
    std::vector<int> result;
    std::vector<int> expected = {2, 4, 1, 3};

    for (int value : data) {
        push_to_list(list, value);
    }

    populate_result(list, result);
    EXPECT_EQ(result, expected);
    free_list(list);
}

TEST(TestList, TestPop) {
    ResourceList list;
    std::vector<int> data = {3, 1, 4, 2, 9, 7};
    std::vector<int> result;
    std::vector<int> expected = {4, 1, 3};

    for (int value : data) {
        push_to_list(list, value);
    }

    for (int i = 0; i < 3; i++) {
        list.pop();
    }

    populate_result(list, result);
    EXPECT_EQ(result, expected);
    free_list(list);
}

TEST(TestList, TestUnlink) {
    ResourceList list;
    std::vector<int> result;
    std::vector<int> expected = {2, 4};

    TestItem *item6 = push_to_list(list, 3);
    TestItem *item5 = push_to_list(list, 1);
    TestItem *item4 = push_to_list(list, 4);
    TestItem *item3 = push_to_list(list, 2);
    TestItem *item2 = push_to_list(list, 9);
    TestItem *item1 = push_to_list(list, 7);
    // 7 9 2 4 1 3

    list.unlink(item2);
    list.unlink(item1);
    list.unlink(item6);
    list.unlink(item5);

    populate_result(list, result);
    EXPECT_EQ(result, expected);
    free_list(list);
}