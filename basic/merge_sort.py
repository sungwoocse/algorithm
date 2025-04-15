def merge_sort(array):
    def _merge(arr, start, mid, end):
        temp = []
        left_index = start
        right_index = mid
        
        while left_index < mid and right_index < end:
            if arr[left_index] <= arr[right_index]:
                temp.append(arr[left_index])
                left_index += 1
            else:
                temp.append(arr[right_index])
                right_index += 1
        
        temp.extend(arr[left_index:mid])
        temp.extend(arr[right_index:end])
        
        arr[start:end] = temp
    
    length = len(array)
    step = 1
    
    while step < length:
        left = 0
        while left < length:
            mid = min(left + step, length)
            right = min(mid + step, length)
            _merge(array, left, mid, right)
            left = right
        step *= 2
    
    return array
