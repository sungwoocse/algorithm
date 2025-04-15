def insertion_sort(array):
    length = len(array)
    i = 1
    
    while i < length:
        key = array[i]
        j = i - 1
        
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1
        
        array[j + 1] = key
        i += 1
    
    return array
