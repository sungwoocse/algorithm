def cocktail_sort(elements):
    result_array = elements.copy()
    total_elements = len(result_array)
    lower_bound = 0
    upper_bound = total_elements - 1
    any_swap_occurred = True
    
    while any_swap_occurred == True and lower_bound < upper_bound:
        any_swap_occurred = False
        
        current_position = lower_bound
        while current_position < upper_bound:
            if result_array[current_position] > result_array[current_position + 1]:
                temp = result_array[current_position]
                result_array[current_position] = result_array[current_position + 1]
                result_array[current_position + 1] = temp
                any_swap_occurred = True
            current_position += 1
        
        if any_swap_occurred == False:
            return result_array
            
        upper_bound -= 1
        any_swap_occurred = False
        
        current_position = upper_bound
        while current_position > lower_bound:
            if result_array[current_position - 1] > result_array[current_position]:
                result_array[current_position - 1], result_array[current_position] = result_array[current_position], result_array[current_position - 1]
                any_swap_occurred = True
            current_position -= 1
            
        lower_bound += 1
    
    return result_array