"""LeetCode 704: Binary Search (Easy)

Given an array of integers nums which is sorted in ascending order,
and an integer target, write a function to search target in nums.
If target exists, then return its index. Otherwise, return -1.

Time Complexity: O(log N)
Space Complexity: O(1)
"""


def search(nums: list[int], target: int) -> int:
    """Finds target index in a sorted list nums using binary search."""
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


if __name__ == "__main__":
    assert search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert search([-1, 0, 3, 5, 9, 12], 2) == -1
    print("LeetCode 704 (Binary Search) passed all test cases!")
