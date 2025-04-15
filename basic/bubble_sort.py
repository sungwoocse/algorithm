def bubble_sort(array):
    size = len(array)
    pass_num = 0
    swapped = True
    
    while swapped and pass_num < size:
        swapped = False
        index = 0
        
        while index < size - pass_num - 1:
            if array[index] > array[index + 1]:
                array[index], array[index + 1] = array[index + 1], array[index]
                swapped = True
            index += 1
        pass_num += 1
    
    return array
