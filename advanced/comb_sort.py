def comb_sort(array):
    result_array = array.copy()
    elements_count = len(result_array)
    current_gap = elements_count 
    
    gap_reduction = 1.25
    has_swapped = True
    
    while current_gap > 1 or has_swapped == True:
        current_gap = max(1, int(current_gap / gap_reduction))
        
        has_swapped = False
        
        position = 0
        while position < elements_count - current_gap:
            if result_array[position] > result_array[position + current_gap]:
                temp = result_array[position]
                result_array[position] = result_array[position + current_gap]
                result_array[position + current_gap] = temp
                has_swapped = True
            position += 1
    
    return result_array