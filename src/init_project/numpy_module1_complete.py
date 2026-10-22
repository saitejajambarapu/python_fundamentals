"""NumPy Module 1: runnable examples from fundamentals to cosine similarity.

Install: python -m pip install numpy
Run:     python numpy_module1_complete.py
"""

import numpy as np


def section(name: str) -> None:
    print(f"\n{'=' * 16} {name} {'=' * 16}")


def main() -> None:
    section('Version and Python lists')
    print('NumPy version:', np.__version__)
    numbers = [10, 20, 30, 40]
    doubled = []
    for number in numbers:
        doubled.append(number * 2)
    print('Python loop:', doubled)
    print('Python list * 2:', [1, 2, 3] * 2)

    section('NumPy arrays and arithmetic')
    values = np.array([10, 20, 30, 40])
    print('Array:', values)
    print('Array * 2:', values * 2)
    print('NumPy [1,2,3] * 2:', np.array([1, 2, 3]) * 2)
    vector = np.array([10, 20, 30])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print('Vector:', vector)
    print('Matrix:\n', matrix)
    print('ndim:', matrix.ndim, 'shape:', matrix.shape,
          'size:', matrix.size, 'dtype:', matrix.dtype)

    section('Creating arrays')
    print('np.array:', np.array([1, 2, 3]))
    print('np.zeros:', np.zeros(4))
    print('np.ones:\n', np.ones((2, 3)))
    print('np.arange:', np.arange(0, 10, 2))
    print('np.linspace:', np.linspace(0, 1, 5))

    section('Arithmetic and aggregations')
    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])
    print('a + b:', a + b)
    print('a - b:', a - b)
    print('a * b:', a * b)
    print('a / b:', a / b)
    print('a ** 2:', a ** 2)
    print('sum:', np.sum(a), 'mean:', np.mean(a))
    print('min:', np.min(a), 'max:', np.max(a))
    print('argmin:', np.argmin(a), 'argmax:', np.argmax(a))
    print('sqrt:', np.sqrt(a))
    print('abs:', np.abs(np.array([-10, 0, 20])))

    section('Reshaping, indexing, and filtering')
    items = np.arange(6)
    grid = items.reshape(2, 3)
    print('reshape:\n', grid)
    print('row 0 col 1:', grid[0, 1])
    print('second row:', grid[1, :])
    print('first column:', grid[:, 0])
    print('flatten:', grid.flatten())
    print('ravel:', grid.ravel())
    print('transpose:\n', grid.T)
    scores = np.array([0.2, 0.9, 0.6, 0.1])
    print('scores >= 0.5:', scores[scores >= 0.5])
    print('sorted score indices:', np.argsort(scores))
    print('concatenate:', np.concatenate([np.array([1, 2]), np.array([3, 4])]))
    print('stack:\n', np.stack([np.array([1, 2]), np.array([3, 4])]))
    print('where:', np.where(scores >= 0.5, 'keep', 'skip'))
    print('clip:', np.clip(scores, 0.3, 0.8))

    section('Dot product, norms, and cosine similarity')
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print('dot:', np.dot(a, b))
    print('a @ b:', a @ b)
    print('norm(a):', np.linalg.norm(a))
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print('cosine similarity:', cosine)
    print('matmul:\n', np.matmul(np.array([[1, 2], [3, 4]]),
                                  np.array([[5, 6], [7, 8]])))

    section('Memory layout and broadcasting')
    memory_array = np.array([10, 20, 30], dtype=np.float32)
    print('dtype:', memory_array.dtype)
    print('shape:', memory_array.shape)
    print('strides:', memory_array.strides)
    print('nbytes:', memory_array.nbytes)
    print('scalar broadcasting:', np.array([1, 2, 3]) + 10)
    grid = np.array([[1, 2, 3], [4, 5, 6]])
    offsets = np.array([10, 20, 30])
    print('row broadcasting:\n', grid + offsets)

    section('Practice answers')
    exercise1 = np.array([5, 10, 15, 20])
    print('Exercise 1 multiplied:', exercise1 * 3)
    print('Exercise 1 mean:', np.mean(exercise1))
    exercise2 = np.arange(1, 13).reshape(3, 4)
    print('Exercise 2:\n', exercise2)
    print('Exercise 2 shape:', exercise2.shape, 'ndim:', exercise2.ndim)
    exercise3_a = np.array([1.0, 0.0])
    exercise3_b = np.array([0.8, 0.6])
    print('Exercise 3 dot:', np.dot(exercise3_a, exercise3_b))
    print('Exercise 3 cosine:', np.dot(exercise3_a, exercise3_b) /
          (np.linalg.norm(exercise3_a) * np.linalg.norm(exercise3_b)))


if __name__ == '__main__':
    main()
