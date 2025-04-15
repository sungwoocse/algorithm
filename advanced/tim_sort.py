def tim_sort(sequence):
  if len(sequence) <= 1:
      return sequence.copy()
  
  sorted_seq = sequence.copy()
  
  min_run_length = calculate_min_run(len(sorted_seq))
  
  start_position = 0
  while start_position < len(sorted_seq):
      end_position = min(start_position + min_run_length, len(sorted_seq))
      sort_run_with_insertion(sorted_seq, start_position, end_position)
      start_position += min_run_length
  
  merge_size = min_run_length
  while merge_size < len(sorted_seq):
      merge_start = 0
      while merge_start < len(sorted_seq):
          merge_mid = min(merge_start + merge_size, len(sorted_seq))
          merge_end = min(merge_start + 2 * merge_size, len(sorted_seq))
          
          if merge_mid < merge_end and sorted_seq[merge_mid - 1] > sorted_seq[merge_mid]:
              merge_runs(sorted_seq, merge_start, merge_mid, merge_end)
          
          merge_start += 2 * merge_size
      
      merge_size *= 2
  
  return sorted_seq

def calculate_min_run(length):
  r = 0
  while length >= 64:
      r |= length & 1
      length >>= 1
  return length + r

def sort_run_with_insertion(sequence, start_position, end_position):
  i = start_position + 1
  while i < end_position:
      current_item = sequence[i]
      position = i - 1
      
      while position >= start_position and sequence[position] > current_item:
          sequence[position + 1] = sequence[position]
          position -= 1
      
      sequence[position + 1] = current_item
      i += 1

def merge_runs(sequence, start_position, mid_position, end_position):
  left_part = []
  i = start_position
  while i < mid_position:
      left_part.append(sequence[i])
      i += 1
  
  right_part = []
  i = mid_position
  while i < end_position:
      right_part.append(sequence[i])
      i += 1
  
  left_index = 0
  right_index = 0
  result_position = start_position
  
  while left_index < len(left_part) and right_index < len(right_part):
      if left_part[left_index] <= right_part[right_index]:
          sequence[result_position] = left_part[left_index]
          left_index += 1
      else:
          sequence[result_position] = right_part[right_index]
          right_index += 1
      result_position += 1
  
  while left_index < len(left_part):
      sequence[result_position] = left_part[left_index]
      left_index += 1
      result_position += 1
  
  for i in range(right_index, len(right_part)):
      sequence[result_position] = right_part[i]
      result_position += 1