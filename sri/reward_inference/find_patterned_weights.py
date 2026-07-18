import numpy as np

# Create the array A
np.random.seed(42)  # Set a seed for reproducibility
A = 2 * np.random.random_sample((10000, 10)) - 1

# Task 1: Find elements of A such that A[j] = bA[i], up to some small error epsilon
def find_elements_bAi(A, i, j, b, epsilon):
    results = np.where(np.abs(A[:, j] - b * A[:, i]) < epsilon)
    return A[results]

# Task 2: Find elements of A such that A[i] = c, up to some small error epsilon
def find_elements_Ai_equals_c(A, i, c, epsilon):
    results = np.where(np.abs(A[:, i] - c) < epsilon)
    return A[results]

# Task 3: Find elements of A such that A[i] is in the range (q, p)
def find_elements_Ai_in_range(A, i, q, p):
    results = np.where((A[:, i] > q) & (A[:, i] < p))
    return A[results]

# Example usage
epsilon = 1e-2
i, j, b = 2, 4, 0.5
c = 0.3
q, p = -0.5, 0.5

task1_result = find_elements_bAi(A, i, j, b, epsilon)
task2_result = find_elements_Ai_equals_c(A, i, c, epsilon)
task3_result = find_elements_Ai_in_range(A, i, q, p)

print("Task 1 result:")
print(task1_result.shape)
print("\nTask 2 result:")
print(task2_result.shape)
print("\nTask 3 result:")
print(task3_result.shape)