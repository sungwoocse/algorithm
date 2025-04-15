function combsort(elements):
    array_size ← length of elements
    interval ← array_size
    reduction_factor ← 1.25
    is_unsorted ← true
    
    while interval > 1 or is_unsorted = true:
        interval ← integer(interval / reduction_factor)
        if interval < 1:
            interval ← 1
        
        is_unsorted ← false

        for position from 0 to array_size-interval-1:
            if elements[position] > elements[position+interval]:
                swap elements[position] and elements[position+interval]
                is_unsorted ← true
    
    return elements