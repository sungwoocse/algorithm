def tournament_sort(elements):
  elements_count = len(elements)
  if elements_count <= 1:
      return elements.copy()

  num_leaves = 1
  while num_leaves < elements_count:
      num_leaves *= 2

  tournament_tree_size = 2 * num_leaves - 1
  tournament_tree = [(float('inf'), -1)] * tournament_tree_size

  element_index = 0
  while element_index < elements_count:
      leaf_position = num_leaves - 1 + element_index
      tournament_tree[leaf_position] = (elements[element_index], element_index)
      element_index += 1

  node_position = num_leaves - 2
  while node_position >= 0:
      left_child_position = 2 * node_position + 1
      right_child_position = 2 * node_position + 2
      
      if tournament_tree[left_child_position][0] <= tournament_tree[right_child_position][0]:
          tournament_tree[node_position] = tournament_tree[left_child_position]
      else:
          tournament_tree[node_position] = tournament_tree[right_child_position]
      
      node_position -= 1
  
  sorted_result = []
  processed_elements = [False] * elements_count

  remaining_count = elements_count
  while remaining_count > 0:
      min_value, min_index = tournament_tree[0]
      sorted_result.append(min_value)
      processed_elements[min_index] = True
      
      leaf_position = num_leaves - 1 + min_index
      tournament_tree[leaf_position] = (float('inf'), -1)

      current_position = leaf_position
      while current_position > 0:
          parent_position = (current_position - 1) // 2
          left_child_position = 2 * parent_position + 1
          right_child_position = 2 * parent_position + 2
          
          left_value = tournament_tree[left_child_position][0]
          right_value = tournament_tree[right_child_position][0]
          
          if left_value <= right_value:
              tournament_tree[parent_position] = tournament_tree[left_child_position]
          else:
              tournament_tree[parent_position] = tournament_tree[right_child_position]
          
          current_position = parent_position
      
      remaining_count -= 1
  
  return sorted_result