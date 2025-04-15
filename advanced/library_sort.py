def library_sort(data):
    if len(data) <= 1:
        return data.copy()
    
    gap_factor = 1.0
    data_size = len(data)
    total_slots = int((1 + gap_factor) * data_size)
    
    sorted_array = [None] * total_slots
    
    sorted_array[0] = data[0]
    filled_elements = 1
    
    min_redistribute_ratio = 0.7
    
    element_index = 1
    while element_index < data_size:
        if filled_elements >= len(sorted_array) * min_redistribute_ratio:
            sorted_array = redistribute_array(sorted_array, filled_elements, gap_factor)
        
        add_element(sorted_array, data[element_index], filled_elements)
        filled_elements += 1
        element_index += 1
    
    final_result = []
    array_index = 0
    while array_index < len(sorted_array):
        if sorted_array[array_index] is not None:
            final_result.append(sorted_array[array_index])
        array_index += 1
        
    return final_result

def add_element(sorted_array, new_value, filled_count):
    insert_position = find_insertion_point(sorted_array, new_value, filled_count)
    
    current_position = insert_position
    while current_position < len(sorted_array):
        if sorted_array[current_position] is None:
            break
        current_position += 1
    
    if current_position >= len(sorted_array):
        current_position = len(sorted_array) - 1
    
    position = current_position
    while position > insert_position:
        sorted_array[position] = sorted_array[position - 1]
        position -= 1
    
    sorted_array[insert_position] = new_value

def find_insertion_point(sorted_array, new_value, filled_count):
    existing_values = []
    for item in sorted_array:
        if item is not None:
            existing_values.append(item)

    for i in range(1, len(existing_values)):
        key = existing_values[i]
        j = i - 1
        while j >= 0 and existing_values[j] > key:
            existing_values[j + 1] = existing_values[j]
            j -= 1
        existing_values[j + 1] = key
    
    start_search = 0
    end_search = len(existing_values) - 1
    
    while start_search <= end_search:
        middle = (start_search + end_search) // 2
        
        if existing_values[middle] < new_value:
            start_search = middle + 1
        else:
            end_search = middle - 1
    
    if start_search == 0:
        for index in range(len(sorted_array)):
            if sorted_array[index] is not None:
                return index
        return 0
    
    if start_search >= len(existing_values):
        last_position = -1
        for index in range(len(sorted_array) - 1, -1, -1):
            if sorted_array[index] is not None:
                last_position = index + 1
                break
        
        if last_position == -1 or last_position >= len(sorted_array):
            return len(sorted_array) - 1
        return last_position
    
    previous_value = existing_values[start_search - 1]
    for index in range(len(sorted_array)):
        if sorted_array[index] == previous_value:
            return index + 1
    
    return 0

def redistribute_array(sorted_array, filled_count, gap_factor):
    values = []
    for item in sorted_array:
        if item is not None:
            values.append(item)

    for i in range(len(values)):
        for j in range(0, len(values) - i - 1):
            if values[j] > values[j + 1]:
                values[j], values[j + 1] = values[j + 1], values[j]
    
    expanded_capacity = int((1 + gap_factor) * (2 * filled_count))
    expanded_array = [None] * expanded_capacity
    
    gap_size = 1 + gap_factor
    
    value_index = 0
    while value_index < len(values):
        position = int(value_index * gap_size)
        if position < len(expanded_array):
            expanded_array[position] = values[value_index]
        value_index += 1
    
    return expanded_array