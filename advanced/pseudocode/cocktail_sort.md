function cocktailsort(elements):
    array_length ← length of elements
    is_sorted ← false
    left_boundary ← 0
    right_boundary ← array_length - 1
    
    while not is_sorted:
        is_sorted ← true

        for current_index from left_boundary to right_boundary-1:
            if elements[current_index] > elements[current_index+1]:
                swap elements[current_index] and elements[current_index+1]
                is_sorted ← false
        
        if is_sorted:
            break
        
        right_boundary ← right_boundary - 1
        
        is_sorted ← true
        
        for current_index from right_boundary-1 down to left_boundary:
            if elements[current_index] > elements[current_index+1]:
                swap elements[current_index] and elements[current_index+1]
                is_sorted ← false
        
        left_boundary ← left_boundary + 1