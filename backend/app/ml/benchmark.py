"""
ml/benchmark.py — Held-out benchmark of real DSA functions.

These are deliberately NOT in the training corpus. They are the honest test:
if the model has learned structure rather than memorised the corpus, it should
get these right without ever having seen them.

Keep it that way. If a benchmark case is added to corpus.py, the benchmark
number stops meaning anything.
"""

from typing import List, Tuple

# (name, expected Big-O label, source)
BENCHMARK: List[Tuple[str, str, str]] = [

 # (name, expected_label, code)
 ("linear_search", "O(n)", """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""),
 ("binary_search_iter", "O(log n)", """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""),
 ("bubble_sort", "O(n²)", """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""),
 ("two_sum_brute", "O(n²)", """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""),
 ("two_sum_hash", "O(n)", """
def two_sum(nums, target):
    seen = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return [seen[target - v], i]
        seen[v] = i
    return []
"""),
 ("fib_naive", "O(2^n)", """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""),
 ("fib_memo", "O(n)", """
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n - 1) + fib(n - 2)
    return memo[n]
"""),
 ("factorial_rec", "O(n)", """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""),
 ("merge_sort", "O(n log n)", """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result
"""),
 ("sort_then_scan", "O(n log n)", """
def has_close_pair(nums, k):
    nums = sorted(nums)
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] < k:
            return True
    return False
"""),
 ("matrix_mult", "O(n³+)", """
def matmul(a, b, n):
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c
"""),
 ("sum_list", "O(n)", """
def total(nums):
    s = 0
    for x in nums:
        s += x
    return s
"""),
 ("two_seq_loops", "O(n)", """
def process(nums):
    total = 0
    for x in nums:
        total += x
    biggest = nums[0]
    for x in nums:
        if x > biggest:
            biggest = x
    return total, biggest
"""),
 ("const_time", "O(1)", """
def area(w, h):
    return w * h
"""),
 ("dict_get", "O(1)", """
def lookup(d, key):
    if key in d:
        return d[key]
    return None
"""),
 ("list_in_loop", "O(n²)", """
def common(list_a, list_b):
    out = []
    for x in list_a:
        if x in list_b:
            out.append(x)
    return out
"""),
 ("quad_range", "O(n²)", """
def print_grid(n):
    for i in range(n * n):
        print(i)
"""),
 ("triple_nested_str", "O(n³+)", """
def count_triples(arr):
    c = 0
    for i in range(len(arr)):
        for j in range(len(arr)):
            for k in range(len(arr)):
                if arr[i] + arr[j] == arr[k]:
                    c += 1
    return c
"""),
 ("quicksort", "O(n log n)", """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
"""),
 ("bfs_graph", "O(n)", """
from collections import deque
def bfs(graph, start):
    visited = set([start])
    q = deque([start])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return order
"""),
 # NOTE: this was originally labelled O(n) — that label was WRONG, and the
 # model flagged it after string-concatenation detection was added. Python
 # strings are immutable, so `out = ch + out` copies the whole accumulated
 # string on every iteration: the loop is quadratic. Left in deliberately as
 # the O(n^2) case it actually is, and as a reminder that a hand-written
 # ground-truth label is itself a thing that can be wrong.
 ("reverse_string_concat", "O(n\u00b2)", """
def reverse(s):
    out = ''
    for ch in s:
        out = ch + out
    return out
"""),

 # The linear way to do the same thing, for contrast.
 ("reverse_string_join", "O(n)", """
def reverse(s):
    pieces = []
    for ch in s:
        pieces.append(ch)
    pieces.reverse()
    return ''.join(pieces)
"""),
 ("selection_sort", "O(n²)", """
def selection_sort(arr):
    for i in range(len(arr)):
        m = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[m]:
                m = j
        arr[i], arr[m] = arr[m], arr[i]
    return arr
"""),
]
