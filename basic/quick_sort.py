def quick_sort(array):
    def partition(arr, low, high):
        mid = (low + high) // 2
        arr[mid], arr[high] = arr[high], arr[mid]
        pivot = arr[high]
        i = low - 1
        
        j = low
        while j < high:
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
            j += 1
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    stack = [(0, len(array)-1)]
    
    while stack:
        low, high = stack.pop()
        if low < high:
            pi = partition(array, low, high)
            stack.append((low, pi-1))
            stack.append((pi+1, high))
    
    return array
