function librarysort(input_array):
    gap_factor ← 1
    total_slots ← (1 + gap_factor) * length(input_array)
    sorted_array ← create empty array of size total_slots
    
    sorted_array[0] ← input_array[0]
    element_count ← 1
    
    for current_index from 1 to length(input_array)-1:
        current_element ← input_array[current_index]

        if element_count >= length(sorted_array) / (1 + gap_factor):
            sorted_array ← redistribute_array(sorted_array, element_count)

        insert_position ← find_insertion_point(sorted_array, current_element)

        move_elements(sorted_array, insert_position)

        sorted_array[insert_position] ← current_element
        element_count ← element_count + 1

    return compact_array(sorted_array)