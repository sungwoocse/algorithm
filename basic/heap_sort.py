def heap_sort(array):
    def heapify(arr, size, root):
        while True:
            left = 2 * root + 1
            right = 2 * root + 2
            largest = root
            
            if left < size and arr[left] > arr[largest]:
                largest = left
            if right < size and arr[right] > arr[largest]:
                largest = right
            
            if largest != root:
                arr[root], arr[largest] = arr[largest], arr[root]
                root = largest
            else:
                break
    
    length = len(array)
    i = length // 2 - 1
    
    while i >= 0:
        heapify(array, length, i)
        i -= 1
    
    last = length - 1
    while last > 0:
        array[last], array[0] = array[0], array[last]
        heapify(array, last, 0)
        last -= 1
    
    return array
