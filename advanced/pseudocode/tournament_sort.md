function tournamentsort(elements):
    array_size ← length of elements
    if array_size ≤ 1:
        return copy of elements
    
    sorted_result ← empty array
    
    num_leaves ← 1
    while num_leaves < array_size:
        num_leaves ← num_leaves × 2
    
    tree_capacity ← 2 × num_leaves - 1
    tournament_tree ← array of size tree_capacity, initialized with (∞, -1) pairs

    for element_index from 0 to array_size-1:
        leaf_position ← num_leaves - 1 + element_index
        tournament_tree[leaf_position] ← (elements[element_index], element_index)

    for node_index from num_leaves-2 down to 0:
        left_node ← 2 × node_index + 1
        right_node ← 2 × node_index + 2
        tournament_tree[node_index] ← min(tournament_tree[left_node], tournament_tree[right_node])
    
    processed_flags ← array of size array_size, all elements initialized to false

    for sort_index from 0 to array_size-1:
        min_value, original_index ← tournament_tree[0]
        append min_value to sorted_result
        processed_flags[original_index] ← true
        
        element_position ← num_leaves - 1 + original_index
        tournament_tree[element_position] ← (∞, -1)
        
        rebuild_tournament(tournament_tree, element_position)
    
    return sorted_result

function rebuild_tournament(tournament_tree, current_position):
    while current_position > 0:
        parent_position ← (current_position - 1) ÷ 2
        left_child ← 2 × parent_position + 1
        right_child ← 2 × parent_position + 2
        
        tournament_tree[parent_position] ← min(tournament_tree[left_child], tournament_tree[right_child])
        current_position ← parent_position