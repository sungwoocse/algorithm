import math

def introsort(elements):
    if len(elements) <= 1:
        return elements.copy()
    
    recursion_limit = 2 * math.floor(math.log2(len(elements))) 
    sorted_elements = elements.copy()
    _introsort_recursive(sorted_elements, 0, len(sorted_elements), recursion_limit)

    return sorted_elements

def _introsort_recursive(elements, begin, end, depth_limit):
    segment_size = end - begin
    
    threshold = 18
    if segment_size <= threshold:
        insertion_sort_segment(elements, begin, end)
        return
    
    if depth_limit == 0:
        heap_sort_segment(elements, begin, end)
        return
    
    pivot_index = partition_elements(elements, begin, end)
    
    _introsort_recursive(elements, begin, pivot_index, depth_limit - 1)
    _introsort_recursive(elements, pivot_index + 1, end, depth_limit - 1)

def partition_elements(elements, begin, end):
    if end - begin > 2:
        mid_index = begin + (end - begin) // 2
        
        if elements[begin] > elements[mid_index]:
            elements[begin], elements[mid_index] = elements[mid_index], elements[begin]
            
        if elements[mid_index] > elements[end - 1]:
            elements[mid_index], elements[end - 1] = elements[end - 1], elements[mid_index]
            
        if elements[begin] > elements[mid_index]:
            elements[begin], elements[mid_index] = elements[mid_index], elements[begin]
            
        elements[begin], elements[mid_index] = elements[mid_index], elements[begin]
    
    pivot = elements[begin]
    left_index = begin + 1
    right_index = end - 1
    
    is_done = False
    while is_done == False:
        while left_index <= right_index and elements[left_index] <= pivot:
            left_index += 1
        
        while left_index <= right_index and elements[right_index] > pivot:
            right_index -= 1
        
        if left_index > right_index:
            is_done = True
        else:
            temp = elements[left_index]
            elements[left_index] = elements[right_index]
            elements[right_index] = temp
    
    elements[begin], elements[right_index] = elements[right_index], elements[begin]
    
    return right_index

def insertion_sort_segment(elements, begin, end):
    i = begin + 1
    while i < end:
        current = elements[i]
        j = i - 1
        
        while j >= begin and elements[j] > current:
            elements[j + 1] = elements[j]
            j -= 1
            
        elements[j + 1] = current
        i += 1

def heap_sort_segment(elements, begin, end):
    segment_size = end - begin
    
    i = begin + segment_size // 2 - 1
    while i >= begin:
        heapify_down(elements, i, begin, end)
        i -= 1
    
    i = end - 1
    while i > begin:
        temp = elements[begin]
        elements[begin] = elements[i]
        elements[i] = temp
        heapify_down(elements, begin, begin, i)
        i -= 1

def heapify_down(elements, root_index, begin, end):
    largest_index = root_index
    
    left_child = 2 * (root_index - begin) + 1 + begin
    right_child = 2 * (root_index - begin) + 2 + begin
    
    if left_child < end:
        if elements[left_child] > elements[largest_index]:
            largest_index = left_child
    
    if right_child < end:
        if elements[right_child] > elements[largest_index]:
            largest_index = right_child
    
    if largest_index != root_index:
        temp = elements[root_index]
        elements[root_index] = elements[largest_index]
        elements[largest_index] = temp
        heapify_down(elements, largest_index, begin, end)