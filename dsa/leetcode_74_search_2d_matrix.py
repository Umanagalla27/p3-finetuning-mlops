"""LeetCode 74: Search a 2D Matrix (Medium)

You are given an m x n integer matrix matrix with the following two properties:
- Each row is sorted in non-decreasing order.
- The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

Core concept:
Treating the M x N matrix as a virtual 1D sorted array of length M * N.
Index mapping:
    row = mid // cols
    col = mid % cols

Time Complexity: O(log(M * N))
Space Complexity: O(1)
"""


def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """Searches for target in an m x n matrix using binary search."""
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, (rows * cols) - 1

    while left <= right:
        mid = left + (right - left) // 2
        row = mid // cols
        col = mid % cols
        val = matrix[row][col]

        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False


if __name__ == "__main__":
    test_matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 60],
    ]
    assert search_matrix(test_matrix, 3) is True
    assert search_matrix(test_matrix, 13) is False
    assert search_matrix([[1]], 1) is True
    assert search_matrix([[1]], 2) is False
    print("LeetCode 74 (Search a 2D Matrix) passed all test cases!")
