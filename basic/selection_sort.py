def selection_sort(array):
    size = len(array)
    current = 0
    
    while current < size:
        min_index = current
        compare = current + 1
        
        while compare < size:
            if array[compare] < array[min_index]:
                min_index = compare
            compare += 1
        
        if min_index != current:
            array[current], array[min_index] = array[min_index], array[current]
        
        current += 1
    
    return array
