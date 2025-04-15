function introsort(elements, begin_index, end_index, recursion_limit):
    subset_size ← end_index - begin_index

    if recursion_limit = 0:
        heap_sort(elements, begin_index, end_index)
        return

    if subset_size <= 18:
        insertion_sort(elements, begin_index, end_index)
        return

    partition_point ← partition(elements, begin_index, end_index)

    introsort(elements, begin_index, partition_point, recursion_limit - 1)
    introsort(elements, partition_point + 1, end_index, recursion_limit - 1)

function intro_sort_driver(elements):
    array_length ← length of elements
    max_recursion_depth ← 2 * log2(array_length)
    introsort(elements, 0, array_length, max_recursion_depth)