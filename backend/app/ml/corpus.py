"""
ml/corpus.py — Labeled training corpus of REAL Python functions.

Why this file exists
--------------------
The previous version of the predictor was trained on 44 hand-typed feature
vectors — literally rows of numbers written by hand, with the label written
by hand next to them. That has three fatal problems:

  1. The vectors were never produced by the feature extractor, so they could
     (and did) drift out of sync with it.
  2. Two different algorithms with the same structure produce the SAME vector.
     A hand-written row for BFS (labeled O(n)) was byte-identical to a row for
     bubble sort (labeled O(n²)). Contradictory labels on identical inputs
     silently destroyed the whole O(n²) class.
  3. Nothing taught the model which features are IRRELEVANT. With one row per
     situation, the forest happily split on statement count or branch count,
     which have nothing to do with asymptotic complexity.

This file fixes all three by storing real Python source. Features are always
extracted by the live extractor at training time, so they cannot drift. Each
sample is real code whose complexity is known because it is a textbook
algorithm. And `_variants()` mutates every sample along dimensions that must
NOT change the answer (variable names, extra statements, extra branches,
for-vs-while), which is what teaches the model to ignore them.

Labels
------
  0 O(1)   1 O(log n)   2 O(n)   3 O(n log n)   4 O(n²)   5 O(n³+)   6 O(2^n)
"""

from __future__ import annotations

import textwrap
from typing import List, Tuple

O1, OLOGN, ON, ONLOGN, ON2, ON3, OEXP = 0, 1, 2, 3, 4, 5, 6

# ---------------------------------------------------------------------------
# Base samples — real algorithms, one entry per (code, label)
# ---------------------------------------------------------------------------

_BASE: List[Tuple[str, int]] = [

    # ── O(1) ───────────────────────────────────────────────────────────────
    ("""
    def rectangle_area(width, height):
        return width * height
    """, O1),

    ("""
    def is_even(n):
        if n % 2 == 0:
            return True
        return False
    """, O1),

    ("""
    def dict_lookup(table, key):
        if key in table:
            return table[key]
        return None
    """, O1),

    ("""
    def swap_first_last(arr):
        if len(arr) < 2:
            return arr
        arr[0], arr[-1] = arr[-1], arr[0]
        return arr
    """, O1),

    ("""
    def classify(score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        else:
            return 'F'
    """, O1),

    ("""
    def stack_push(stack, value):
        stack.append(value)
        return len(stack)
    """, O1),

    ("""
    def manhattan(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])
    """, O1),

    ("""
    def counter_increment(counts, key):
        if key in counts:
            counts[key] = counts[key] + 1
        else:
            counts[key] = 1
        return counts
    """, O1),

    # ── O(log n) ───────────────────────────────────────────────────────────
    ("""
    def binary_search(arr, target):
        lo = 0
        hi = len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1
    """, OLOGN),

    ("""
    def binary_search_recursive(arr, target, lo, hi):
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            return binary_search_recursive(arr, target, mid + 1, hi)
        return binary_search_recursive(arr, target, lo, mid - 1)
    """, OLOGN),

    ("""
    def count_digits(n):
        digits = 0
        while n > 0:
            n = n // 10
            digits = digits + 1
        return digits
    """, OLOGN),

    ("""
    def fast_power(base, exp):
        result = 1
        while exp > 0:
            if exp % 2 == 1:
                result = result * base
            base = base * base
            exp = exp // 2
        return result
    """, OLOGN),

    ("""
    def height_of_perfect_tree(n):
        h = 0
        while n > 1:
            n = n // 2
            h = h + 1
        return h
    """, OLOGN),

    ("""
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    """, OLOGN),

    # ── O(n) ───────────────────────────────────────────────────────────────
    ("""
    def linear_search(arr, target):
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    """, ON),

    ("""
    def find_max(nums):
        best = nums[0]
        for x in nums:
            if x > best:
                best = x
        return best
    """, ON),

    ("""
    def total(nums):
        s = 0
        for x in nums:
            s = s + x
        return s
    """, ON),

    ("""
    def two_sum_hashmap(nums, target):
        seen = {}
        for i, value in enumerate(nums):
            complement = target - value
            if complement in seen:
                return [seen[complement], i]
            seen[value] = i
        return []
    """, ON),

    ("""
    def count_frequencies(items):
        counts = {}
        for item in items:
            if item in counts:
                counts[item] = counts[item] + 1
            else:
                counts[item] = 1
        return counts
    """, ON),

    ("""
    def has_duplicate(nums):
        seen = set()
        for x in nums:
            if x in seen:
                return True
            seen.add(x)
        return False
    """, ON),

    ("""
    def reverse_list(arr):
        out = []
        for i in range(len(arr) - 1, -1, -1):
            out.append(arr[i])
        return out
    """, ON),

    ("""
    def running_total(nums):
        out = []
        acc = 0
        for x in nums:
            acc = acc + x
            out.append(acc)
        return out
    """, ON),

    ("""
    def sum_then_max(nums):
        s = 0
        for x in nums:
            s = s + x
        best = nums[0]
        for x in nums:
            if x > best:
                best = x
        return s, best
    """, ON),

    ("""
    def three_sequential_passes(nums):
        a = 0
        for x in nums:
            a = a + x
        b = 0
        for x in nums:
            b = b + x * x
        c = 0
        for x in nums:
            if x > 0:
                c = c + 1
        return a, b, c
    """, ON),

    ("""
    def squares(nums):
        return [x * x for x in nums]
    """, ON),

    ("""
    def positives(nums):
        return [x for x in nums if x > 0]
    """, ON),

    ("""
    def build_index(words):
        return {w: i for i, w in enumerate(words)}
    """, ON),

    ("""
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    """, ON),

    ("""
    def list_sum_recursive(arr, i):
        if i >= len(arr):
            return 0
        return arr[i] + list_sum_recursive(arr, i + 1)
    """, ON),

    ("""
    def fib_memo(n, memo):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
        return memo[n]
    """, ON),

    ("""
    def climb_stairs_memo(n, cache):
        if n in cache:
            return cache[n]
        if n <= 2:
            return n
        cache[n] = climb_stairs_memo(n - 1, cache) + climb_stairs_memo(n - 2, cache)
        return cache[n]
    """, ON),

    ("""
    def bfs(graph, start):
        visited = set()
        visited.add(start)
        queue = [start]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return order
    """, ON),

    ("""
    def dfs_iterative(graph, start):
        seen = set()
        stack = [start]
        result = []
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                result.append(node)
                for nb in graph[node]:
                    stack.append(nb)
        return result
    """, ON),

    ("""
    def max_subarray(nums):
        best = nums[0]
        current = nums[0]
        for x in nums[1:]:
            current = max(x, current + x)
            best = max(best, current)
        return best
    """, ON),

    ("""
    def is_palindrome(s):
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left = left + 1
            right = right - 1
        return True
    """, ON),

    ("""
    def merge_two_sorted(a, b):
        out = []
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                out.append(a[i])
                i = i + 1
            else:
                out.append(b[j])
                j = j + 1
        return out
    """, ON),

    # ── O(n log n) ─────────────────────────────────────────────────────────
    ("""
    def sort_numbers(nums):
        return sorted(nums)
    """, ONLOGN),

    ("""
    def sort_in_place(nums):
        nums.sort()
        return nums
    """, ONLOGN),

    ("""
    def has_close_pair(nums, k):
        nums = sorted(nums)
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] < k:
                return True
        return False
    """, ONLOGN),

    ("""
    def top_k(nums, k):
        ordered = sorted(nums, reverse=True)
        out = []
        for i in range(k):
            out.append(ordered[i])
        return out
    """, ONLOGN),

    ("""
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i = i + 1
            else:
                result.append(right[j])
                j = j + 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    """, ONLOGN),

    ("""
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        smaller = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        larger = [x for x in arr if x > pivot]
        return quicksort(smaller) + equal + quicksort(larger)
    """, ONLOGN),

    ("""
    def sort_by_frequency(items):
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return sorted(items, key=lambda x: counts[x])
    """, ONLOGN),

    ("""
    def anagram_groups(words):
        groups = {}
        for w in words:
            key = ''.join(sorted(w))
            if key in groups:
                groups[key].append(w)
            else:
                groups[key] = [w]
        return groups
    """, ONLOGN),

    ("""
    def median_of_list(nums):
        ordered = sorted(nums)
        mid = len(ordered) // 2
        return ordered[mid]
    """, ONLOGN),

    # ── O(n²) ──────────────────────────────────────────────────────────────
    ("""
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    """, ON2),

    ("""
    def selection_sort(arr):
        for i in range(len(arr)):
            smallest = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[smallest]:
                    smallest = j
            arr[i], arr[smallest] = arr[smallest], arr[i]
        return arr
    """, ON2),

    ("""
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j = j - 1
            arr[j + 1] = key
        return arr
    """, ON2),

    ("""
    def two_sum_bruteforce(nums, target):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
    """, ON2),

    ("""
    def find_duplicates_bruteforce(nums):
        duplicates = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j]:
                    duplicates.append(nums[i])
        return duplicates
    """, ON2),

    ("""
    def common_elements_list(list_a, list_b):
        out = []
        for x in list_a:
            if x in list_b:
                out.append(x)
        return out
    """, ON2),

    ("""
    def remove_duplicates_list(nums):
        out = []
        for x in nums:
            if x not in out:
                out.append(x)
        return out
    """, ON2),

    ("""
    def print_grid(n):
        for i in range(n * n):
            print(i)
    """, ON2),

    ("""
    def pair_products(nums):
        return [x * y for x in nums for y in nums]
    """, ON2),

    ("""
    def transpose_and_sum(matrix, n):
        total = 0
        for i in range(n):
            for j in range(n):
                total = total + matrix[i][j]
        return total
    """, ON2),

    ("""
    def longest_common_prefix_pairs(words):
        best = 0
        for a in words:
            for b in words:
                if a != b and a[0] == b[0]:
                    best = best + 1
        return best
    """, ON2),

    ("""
    def count_inversions_bruteforce(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    count = count + 1
        return count
    """, ON2),

    ("""
    def build_pairs(nums):
        pairs = []
        i = 0
        while i < len(nums):
            j = 0
            while j < len(nums):
                pairs.append((nums[i], nums[j]))
                j = j + 1
            i = i + 1
        return pairs
    """, ON2),

    ("""
    def index_lookup_in_loop(items, targets):
        positions = []
        for t in targets:
            positions.append(items.index(t))
        return positions
    """, ON2),

    # ── O(n³+) ─────────────────────────────────────────────────────────────
    ("""
    def matrix_multiply(a, b, n):
        c = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c[i][j] = c[i][j] + a[i][k] * b[k][j]
        return c
    """, ON3),

    ("""
    def three_sum_bruteforce(nums, target):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    if nums[i] + nums[j] + nums[k] == target:
                        return [i, j, k]
        return []
    """, ON3),

    ("""
    def count_triples(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(len(arr)):
                for k in range(len(arr)):
                    if arr[i] + arr[j] == arr[k]:
                        count = count + 1
        return count
    """, ON3),

    ("""
    def floyd_warshall(dist, n):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist
    """, ON3),

    ("""
    def cube_sum(n):
        total = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    total = total + i * j * k
        return total
    """, ON3),

    ("""
    def quad_nested(n):
        count = 0
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        count = count + 1
        return count
    """, ON3),

    ("""
    def triple_comprehension(nums):
        return [x + y + z for x in nums for y in nums for z in nums]
    """, ON3),

    # ── O(2^n) ─────────────────────────────────────────────────────────────
    ("""
    def fib_naive(n):
        if n <= 1:
            return n
        return fib_naive(n - 1) + fib_naive(n - 2)
    """, OEXP),

    ("""
    def climb_stairs_naive(n):
        if n <= 2:
            return n
        return climb_stairs_naive(n - 1) + climb_stairs_naive(n - 2)
    """, OEXP),

    ("""
    def count_paths(x, y):
        if x == 0 or y == 0:
            return 1
        return count_paths(x - 1, y) + count_paths(x, y - 1)
    """, OEXP),

    ("""
    def subset_sum(nums, i, target):
        if target == 0:
            return True
        if i >= len(nums):
            return False
        return subset_sum(nums, i + 1, target - nums[i]) or subset_sum(nums, i + 1, target)
    """, OEXP),

    ("""
    def knapsack_naive(weights, values, i, capacity):
        if i >= len(weights) or capacity <= 0:
            return 0
        skip = knapsack_naive(weights, values, i + 1, capacity)
        take = values[i] + knapsack_naive(weights, values, i + 1, capacity - weights[i])
        if take > skip:
            return take
        return skip
    """, OEXP),

    ("""
    def tribonacci_naive(n):
        if n < 3:
            return 1
        return tribonacci_naive(n - 1) + tribonacci_naive(n - 2) + tribonacci_naive(n - 3)
    """, OEXP),

    ("""
    def hanoi_moves(n):
        if n <= 1:
            return 1
        return hanoi_moves(n - 1) + 1 + hanoi_moves(n - 1)
    """, OEXP),

    ("""
    def lcs_naive(a, b, i, j):
        if i >= len(a) or j >= len(b):
            return 0
        if a[i] == b[j]:
            return 1 + lcs_naive(a, b, i + 1, j + 1)
        left = lcs_naive(a, b, i + 1, j)
        right = lcs_naive(a, b, i, j + 1)
        if left > right:
            return left
        return right
    """, OEXP),
    # ─────────────────────────────────────────────────────────────────────
    # Expansion round 2 — patterns users actually paste
    # (string work, matrices, linked lists, sliding window, two pointers,
    #  heaps, and more O(log n), which was the weakest class at 83%)
    # ─────────────────────────────────────────────────────────────────────

    # ── O(log n) — the class the confusion matrix flagged ─────────────────
    ("""
    def count_set_bits(n):
        count = 0
        while n > 0:
            count = count + (n & 1)
            n = n >> 1
        return count
    """, OLOGN),

    ("""
    def integer_sqrt(n):
        lo = 0
        hi = n
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid * mid == n:
                return mid
            if mid * mid < n:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi
    """, OLOGN),

    ("""
    def first_occurrence(arr, target):
        lo = 0
        hi = len(arr) - 1
        found = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                found = mid
                hi = mid - 1
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return found
    """, OLOGN),

    ("""
    def search_rotated(arr, target):
        lo = 0
        hi = len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                return mid
            if arr[lo] <= arr[mid]:
                if arr[lo] <= target < arr[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:
                if arr[mid] < target <= arr[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return -1
    """, OLOGN),

    ("""
    def bst_search(node, key):
        if node is None:
            return None
        if node.key == key:
            return node
        if key < node.key:
            return bst_search(node.left, key)
        return bst_search(node.right, key)
    """, OLOGN),

    ("""
    def halve_until_one(n):
        steps = 0
        while n > 1:
            n = n / 2
            steps = steps + 1
        return steps
    """, OLOGN),

    # ── O(n) — strings, linked lists, sliding window, two pointers ────────
    ("""
    def count_vowels(text):
        vowels = set('aeiou')
        total = 0
        for ch in text:
            if ch in vowels:
                total = total + 1
        return total
    """, ON),

    ("""
    def reverse_words_in_place(words):
        left = 0
        right = len(words) - 1
        while left < right:
            words[left], words[right] = words[right], words[left]
            left = left + 1
            right = right - 1
        return words
    """, ON),

    ("""
    def max_window_sum(nums, k):
        window = 0
        for i in range(k):
            window = window + nums[i]
        best = window
        for i in range(k, len(nums)):
            window = window + nums[i] - nums[i - k]
            if window > best:
                best = window
        return best
    """, ON),

    ("""
    def longest_unique_substring(text):
        last_seen = {}
        start = 0
        best = 0
        for i, ch in enumerate(text):
            if ch in last_seen and last_seen[ch] >= start:
                start = last_seen[ch] + 1
            last_seen[ch] = i
            if i - start + 1 > best:
                best = i - start + 1
        return best
    """, ON),

    ("""
    def linked_list_length(head):
        count = 0
        node = head
        while node is not None:
            count = count + 1
            node = node.next
        return count
    """, ON),

    ("""
    def reverse_linked_list(head):
        previous = None
        current = head
        while current is not None:
            nxt = current.next
            current.next = previous
            previous = current
            current = nxt
        return previous
    """, ON),

    ("""
    def find_middle_node(head):
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
        return slow
    """, ON),

    ("""
    def is_balanced_brackets(text):
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in text:
            if ch in '([{':
                stack.append(ch)
            elif ch in pairs:
                if not stack or stack.pop() != pairs[ch]:
                    return False
        return not stack
    """, ON),

    ("""
    def char_frequency(text):
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        return counts
    """, ON),

    ("""
    def two_pointer_pair_sum(sorted_nums, target):
        left = 0
        right = len(sorted_nums) - 1
        while left < right:
            total = sorted_nums[left] + sorted_nums[right]
            if total == target:
                return (left, right)
            if total < target:
                left = left + 1
            else:
                right = right - 1
        return None
    """, ON),

    ("""
    def flatten_one_level(rows):
        flat = []
        for row in rows:
            flat.extend(row)
        return flat
    """, ON),

    ("""
    def fizzbuzz(n):
        out = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                out.append('FizzBuzz')
            elif i % 3 == 0:
                out.append('Fizz')
            else:
                out.append(str(i))
        return out
    """, ON),

    # ── O(n log n) — heaps and sorting-based pipelines ────────────────────
    ("""
    def k_largest(nums, k):
        import heapq
        heap = []
        for x in nums:
            heapq.heappush(heap, x)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap
    """, ONLOGN),

    ("""
    def sort_words_by_length(words):
        return sorted(words, key=len)
    """, ONLOGN),

    ("""
    def dedupe_and_sort(nums):
        unique = set(nums)
        return sorted(unique)
    """, ONLOGN),

    ("""
    def merge_intervals(intervals):
        intervals = sorted(intervals)
        merged = []
        for interval in intervals:
            if merged and interval[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], interval[1])
            else:
                merged.append(interval)
        return merged
    """, ONLOGN),

    ("""
    def count_inversions(nums):
        if len(nums) <= 1:
            return nums, 0
        mid = len(nums) // 2
        left, left_inv = count_inversions(nums[:mid])
        right, right_inv = count_inversions(nums[mid:])
        merged = []
        i = 0
        j = 0
        inversions = left_inv + right_inv
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i = i + 1
            else:
                merged.append(right[j])
                j = j + 1
                inversions = inversions + (len(left) - i)
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions
    """, ONLOGN),

    ("""
    def heap_sort(nums):
        import heapq
        heap = list(nums)
        heapq.heapify(heap)
        ordered = []
        while heap:
            ordered.append(heapq.heappop(heap))
        return ordered
    """, ONLOGN),

    ("""
    def k_smallest(nums, k):
        import heapq
        return heapq.nsmallest(k, nums)
    """, ONLOGN),

    ("""
    def top_k_frequent(items, k):
        import heapq
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return heapq.nlargest(k, counts, key=counts.get)
    """, ONLOGN),

    ("""
    def merge_k_sorted_lists(lists):
        import heapq
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        merged = []
        while heap:
            val, i, j = heapq.heappop(heap)
            merged.append(val)
            if j + 1 < len(lists[i]):
                heapq.heappush(heap, (lists[i][j + 1], i, j + 1))
        return merged
    """, ONLOGN),

    ("""
    def schedule_max_meetings(intervals):
        intervals = sorted(intervals, key=lambda pair: pair[1])
        count = 0
        last_end = float('-inf')
        for start, end in intervals:
            if start >= last_end:
                count = count + 1
                last_end = end
        return count
    """, ONLOGN),

    ("""
    def rank_queries(nums, queries):
        import bisect
        ordered = sorted(nums)
        ranks = []
        for q in queries:
            idx = bisect.bisect_left(ordered, q)
            ranks.append(idx)
        return ranks
    """, ONLOGN),

    ("""
    def closest_pair_gap(points):
        points = sorted(points)
        best = float('inf')
        for i in range(len(points) - 1):
            gap = points[i + 1] - points[i]
            if gap < best:
                best = gap
        return best
    """, ONLOGN),

    # ── O(n²) — matrices and string pair work ────────────────────────────
    ("""
    def rotate_matrix(matrix, n):
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix
    """, ON2),

    ("""
    def longest_palindromic_substring(text):
        best = ''
        for i in range(len(text)):
            for j in range(i, len(text)):
                piece = text[i:j + 1]
                if piece == piece[::-1] and len(piece) > len(best):
                    best = piece
        return best
    """, ON2),

    ("""
    def pairwise_distances(points):
        distances = []
        for a in points:
            for b in points:
                distances.append(abs(a - b))
        return distances
    """, ON2),

    ("""
    def spiral_sum(matrix, n):
        total = 0
        for i in range(n):
            for j in range(n):
                if i == j or i + j == n - 1:
                    total = total + matrix[i][j]
        return total
    """, ON2),

    ("""
    def anagram_pairs(words):
        pairs = 0
        for a in words:
            for b in words:
                if sorted(a) == sorted(b):
                    pairs = pairs + 1
        return pairs
    """, ON2),

    # ── O(n³+) ────────────────────────────────────────────────────────────
    ("""
    def all_triplets(nums):
        triplets = []
        for a in nums:
            for b in nums:
                for c in nums:
                    triplets.append((a, b, c))
        return triplets
    """, ON3),

    ("""
    def matrix_chain_bruteforce(dims, n):
        best = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cost = dims[i] * dims[j] * dims[k]
                    if cost > best:
                        best = cost
        return best
    """, ON3),

    ("""
    def gaussian_eliminate(matrix, n):
        for pivot in range(n):
            for row in range(pivot + 1, n):
                factor = matrix[row][pivot] / matrix[pivot][pivot]
                for col in range(pivot, n):
                    matrix[row][col] = matrix[row][col] - factor * matrix[pivot][col]
        return matrix
    """, ON3),

    ("""
    def count_triangles(adj, n):
        triangles = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if adj[i][j] and adj[j][k] and adj[k][i]:
                        triangles = triangles + 1
        return triangles
    """, ON3),

    ("""
    def common_triplets(list_a, list_b, list_c):
        common = []
        for x in list_a:
            for y in list_b:
                for z in list_c:
                    if x == y == z:
                        common.append((x, y, z))
        return common
    """, ON3),

    ("""
    def gram_matrix(matrix, n):
        result = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i][j] = sum(matrix[i][k] * matrix[j][k] for k in range(n))
        return result
    """, ON3),

    ("""
    def longest_common_substring_bruteforce(a, b):
        best = ''
        for i in range(len(a)):
            for j in range(len(b)):
                k = 0
                while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                    k = k + 1
                if k > len(best):
                    best = a[i:i + k]
        return best
    """, ON3),

    ("""
    def matrix_chain_order(dims, n):
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                dp[i][j] = None
                for k in range(i + 1, j):
                    cost = dp[i][k] + dp[k][j] + dims[i] * dims[k] * dims[j]
                    if dp[i][j] is None or cost < dp[i][j]:
                        dp[i][j] = cost
        return dp
    """, ON3),

    ("""
    def max_subarray_sum_bruteforce(nums):
        n = len(nums)
        best = nums[0]
        for start in range(n):
            for end in range(start, n):
                total = 0
                for k in range(start, end + 1):
                    total = total + nums[k]
                if total > best:
                    best = total
        return best
    """, ON3),

    ("""
    def while_triple_nested(n):
        total = 0
        i = 0
        while i < n:
            j = 0
            while j < n:
                k = 0
                while k < n:
                    total = total + 1
                    k = k + 1
                j = j + 1
            i = i + 1
        return total
    """, ON3),

    ("""
    def bellman_ford(adj, n, source):
        dist = [float('inf')] * n
        dist[source] = 0
        changed = True
        while changed == True:
            changed = False
            for u in range(n):
                for v in range(n):
                    if adj[u][v] is not None and dist[u] + adj[u][v] < dist[v]:
                        dist[v] = dist[u] + adj[u][v]
                        changed = True
        return dist
    """, ON3),

    # ── O(2^n) ────────────────────────────────────────────────────────────
    ("""
    def all_subsets(nums, i):
        if i >= len(nums):
            return 1
        with_item = all_subsets(nums, i + 1)
        without_item = all_subsets(nums, i + 1)
        return with_item + without_item
    """, OEXP),

    ("""
    def partition_count(n, k):
        if n == 0:
            return 1
        if n < 0 or k <= 0:
            return 0
        return partition_count(n - k, k) + partition_count(n, k - 1)
    """, OEXP),

    ("""
    def edit_distance_naive(a, b, i, j):
        if i >= len(a):
            return len(b) - j
        if j >= len(b):
            return len(a) - i
        if a[i] == b[j]:
            return edit_distance_naive(a, b, i + 1, j + 1)
        insert_cost = edit_distance_naive(a, b, i, j + 1)
        delete_cost = edit_distance_naive(a, b, i + 1, j)
        return 1 + min(insert_cost, delete_cost)
    """, OEXP),
    # ─────────────────────────────────────────────────────────────────────
    # Expansion round 3 — patterns the audit caught the model getting wrong
    # ─────────────────────────────────────────────────────────────────────

    # ── String concatenation in a loop is QUADRATIC ──────────────────────
    # Strings are immutable, so `out = out + piece` copies everything
    # accumulated so far on every iteration. Looks linear, is not.
    ("""
    def join_parts(parts):
        out = ''
        for p in parts:
            out = out + p
        return out
    """, ON2),

    ("""
    def build_csv_row(values):
        line = ''
        for v in values:
            line += str(v) + ','
        return line
    """, ON2),

    ("""
    def reverse_string_slow(text):
        out = ''
        for ch in text:
            out = ch + out
        return out
    """, ON2),

    ("""
    def repeat_label(word, times):
        banner = ''
        for i in range(times):
            banner = banner + word
        return banner
    """, ON2),

    # The same SHAPE on a number is genuinely linear — this pair is what
    # teaches the model to use the inferred type, not the syntax.
    ("""
    def sum_values(nums):
        running = 0
        for x in nums:
            running = running + x
        return running
    """, ON),

    ("""
    def accumulate_counts(nums):
        tally = 0
        for x in nums:
            tally += x
        return tally
    """, ON),

    # ── A sort INSIDE a loop runs n times ────────────────────────────────
    ("""
    def sort_each_group(groups):
        out = []
        for group in groups:
            out.append(sorted(group))
        return out
    """, ON2),

    ("""
    def normalise_rows(rows):
        result = []
        for row in rows:
            row.sort()
            result.append(row)
        return result
    """, ON2),

    ("""
    def rank_within_buckets(buckets):
        ranked = {}
        for key in buckets:
            ranked[key] = sorted(buckets[key])
        return ranked
    """, ON2),

    ("""
    def best_of_each(groups):
        winners = []
        for group in groups:
            ordered = sorted(group)
            winners.append(ordered[0])
        return winners
    """, ON2),

    # ── Mutual recursion — recursion with no self-call ───────────────────
    ("""
    def is_even_number(n):
        if n == 0:
            return True
        return is_odd_number(n - 1)

    def is_odd_number(n):
        if n == 0:
            return False
        return is_even_number(n - 1)
    """, ON),

    ("""
    def walk_down(n):
        if n <= 0:
            return 0
        return 1 + walk_up(n - 1)

    def walk_up(n):
        if n <= 0:
            return 0
        return 1 + walk_down(n - 1)
    """, ON),

    ("""
    def branch_a(n):
        if n <= 1:
            return 1
        return branch_b(n - 1) + branch_b(n - 2)

    def branch_b(n):
        if n <= 1:
            return 1
        return branch_a(n - 1) + branch_a(n - 2)
    """, OEXP),

    ("""
    def expand_left(n):
        if n <= 0:
            return 1
        return expand_right(n - 1) + expand_right(n - 1)

    def expand_right(n):
        if n <= 0:
            return 1
        return expand_left(n - 1) + expand_left(n - 1)
    """, OEXP),
    # ── Whole-collection work with no visible loop is still O(n) ─────────
    ("""
    def shared_items(first, second):
        return set(first).intersection(set(second))
    """, ON),

    ("""
    def merge_tags(a, b):
        return set(a).union(set(b))
    """, ON),

    ("""
    def join_with_commas(words):
        return ','.join(words)
    """, ON),

    ("""
    def as_sentence(words):
        return ' '.join(words) + '.'
    """, ON),

    ("""
    def copy_and_extend(base, extra):
        result = list(base)
        result.extend(extra)
        return result
    """, ON),

    ("""
    def largest_value(values):
        return max(values)
    """, ON),

    ("""
    def unique_count(values):
        return len(set(values))
    """, ON),

    ("""
    def tail_slice(values):
        return values[1:]
    """, ON),

    # Controls: genuinely O(1) despite touching a collection
    ("""
    def size_of(values):
        return len(values)
    """, O1),

    ("""
    def push_item(stack, value):
        stack.append(value)
        return stack
    """, O1),

    ("""
    def larger_of(a, b):
        return max(a, b)
    """, O1),

    ("""
    def fetch(mapping, key):
        return mapping.get(key)
    """, O1),
]


# ---------------------------------------------------------------------------
# Variant generation — teaches the model what does NOT matter
# ---------------------------------------------------------------------------

_NOISE_STATEMENTS = [
    "    debug_flag = False",
    "    label = 'result'",
    "    counter_offset = 0",
    "    print('starting')",
    "    scratch = []",
]

_NOISE_BRANCH = (
    "    if debug_flag:\n"
    "        pass"
)


def _dedent(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def _inject_after_signature(code: str, extra: str) -> str:
    """Insert `extra` immediately after the first `def ...:` line."""
    lines = code.split("\n")
    for idx, line in enumerate(lines):
        if line.rstrip().endswith(":") and line.lstrip().startswith("def "):
            return "\n".join(lines[: idx + 1] + [extra] + lines[idx + 1:])
    return code


def _rename_identifiers(code: str) -> str:
    """Crude but safe-enough renaming for training variety."""
    replacements = [
        ("arr", "sequence"), ("nums", "values"), ("out", "collected"),
        ("total", "running_sum"), ("count", "tally"), ("result", "output"),
    ]
    for old, new in replacements:
        code = code.replace(old, new)
    return code


def _variants(code: str) -> List[str]:
    """
    Produce structural no-ops of a sample.

    Every variant here has IDENTICAL asymptotic complexity to the original.
    Their whole job is to vary the features that must not influence the
    prediction — statement count, branch count, identifier names — so the
    forest learns to split on loop structure and recursion instead.
    """
    base = _dedent(code)
    variants = [base]

    # +1 irrelevant statement
    variants.append(_dedent(_inject_after_signature(base, _NOISE_STATEMENTS[0])))

    # +3 irrelevant statements (pushes statement count well up)
    bulk = "\n".join(_NOISE_STATEMENTS[:3])
    variants.append(_dedent(_inject_after_signature(base, bulk)))

    # +1 irrelevant branch (pushes branch count up)
    with_branch = _inject_after_signature(base, _NOISE_STATEMENTS[0])
    with_branch = _inject_after_signature(with_branch, _NOISE_BRANCH)
    variants.append(_dedent(with_branch))

    # renamed identifiers
    variants.append(_dedent(_rename_identifiers(base)))

    return variants


def load_corpus(with_variants: bool = True) -> Tuple[List[str], List[int]]:
    """
    Return (sources, labels).

    Args:
        with_variants: if True, expand each base sample into structural
                       no-op variants. Set False to inspect the base set.
    """
    sources: List[str] = []
    labels: List[int] = []

    for code, label in _BASE:
        if with_variants:
            for v in _variants(code):
                sources.append(v)
                labels.append(label)
        else:
            sources.append(_dedent(code))
            labels.append(label)

    return sources, labels


def class_distribution() -> dict:
    """Sample count per class — used by the eval harness to check balance."""
    from collections import Counter
    _, labels = load_corpus()
    names = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³+)", "O(2^n)"]
    counts = Counter(labels)
    return {names[k]: counts[k] for k in sorted(counts)}


if __name__ == "__main__":
    srcs, labs = load_corpus()
    print(f"corpus: {len(srcs)} samples from {len(_BASE)} base algorithms")
    for name, n in class_distribution().items():
        print(f"  {name:12} {n:4}")
