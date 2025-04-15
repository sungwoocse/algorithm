function timsort(sequence):
    seq_length ← length(sequence)
    
    min_run ← compute_min_run(seq_length)

    for run_start from 0 to seq_length-1 step min_run:
        run_end ← min(run_start + min_run, seq_length)
        sort_small_run(sequence, run_start, run_end)

    merge_size ← min_run
    while merge_size < seq_length:
        for merge_start from 0 to seq_length-1 step 2*merge_size:
            merge_mid ← min(merge_start + merge_size, seq_length)
            merge_end ← min(merge_start + 2*merge_size, seq_length)
            
            if merge_mid < merge_end:
                merge_adjacent_runs(sequence, merge_start, merge_mid, merge_end)
        
        merge_size ← 2 * merge_size
    
    return sequence

function compute_min_run(n):
    r ← 0
    while n >= 64:
        r |= n & 1
        n ← n >> 1
    return n + r

function sort_small_run(sequence, start, end):
    for i from start+1 to end-1:
        current ← sequence[i]
        j ← i - 1
        
        while j >= start and sequence[j] > current:
            sequence[j+1] ← sequence[j]
            j ← j - 1
            
        sequence[j+1] ← current

function merge_adjacent_runs(sequence, start, mid, end):
    if sequence[mid-1] <= sequence[mid]:
        return
    
    left ← copy of sequence[start:mid]
    right ← copy of sequence[mid:end]
    
    i ← 0
    j ← 0
    k ← start
    
    while i < length(left) and j < length(right):
        if left[i] <= right[j]:
            sequence[k] ← left[i]
            i ← i + 1
        else:
            sequence[k] ← right[j]
            j ← j + 1
        
        k ← k + 1
    
    while i < length(left):
        sequence[k] ← left[i]
        k ← k + 1
        i ← i + 1
    
    while j < length(right):
        sequence[k] ← right[j]
        k ← k + 1
        j ← j + 1
