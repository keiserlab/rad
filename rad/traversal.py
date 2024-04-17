from tqdm import tqdm
import heapq

def initialize_search(hnsw_graphs, scoring_fn):
    open_pq = []
    visited = set()
    node_scores = {}
    
    # Evaluate the entire top HNSW layer
    for node in tqdm(hnsw_graphs[-1].vertices(), desc="Initializing Traversal", total=hnsw_graphs[-1].num_vertices()):
        node_id = int(node)
        
        # Graph tools is kind of stupid and keeps track of all nodes so we need
        # to filter just those with neighbors
        if len(hnsw_graphs[-1].get_all_neighbors(node_id)) == 0:
            continue

        score = scoring_fn(node_id)
            
        node_scores[node_id] = score
            
        visited.add((node_id,len(hnsw_graphs)-1))
        heapq.heappush(open_pq, (score, (node_id, len(hnsw_graphs)-1)))
        
    return (open_pq,
            visited,
            node_scores)

def traverse_queue(
        open_pq,
        visited,
        hnsw_graphs,
        max_to_evaluate,
        node_scores,
        scoring_fn):
    
    scores_evaluated = len(node_scores)    

    pbar = tqdm(desc="Traversing HNSW", total=max_to_evaluate)

    while scores_evaluated < max_to_evaluate:
        
        # This shouldn't happen but I have a check
        if len(open_pq) == 0:
            print("Priority Queue empty")
            pbar.close()
            return node_scores
        
        # Pop the **LOWEST** scoring node from the priority queue
        score, (current_node, current_level) = heapq.heappop(open_pq)
        
        for neighbor in hnsw_graphs[current_level].get_all_neighbors(current_node):
            neighbor_id = int(neighbor)
            
            # Skip the neighbor if its been visited
            if (neighbor_id, current_level) in visited:
                continue

            # Score the neighbor (or load its score if previously scored in a different layer)
            if neighbor_id not in node_scores:
                neighbor_score = scoring_fn(neighbor_id)
                node_scores[neighbor_id] = neighbor_score
                scores_evaluated += 1
                pbar.update(1)
                if scores_evaluated >= max_to_evaluate:
                    pbar.close()
                    return node_scores
            else:
                neighbor_score = node_scores[neighbor_id]
                
            visited.add((neighbor_id, current_level))
            heapq.heappush(open_pq, (neighbor_score, (neighbor_id, current_level)))
            
            
        # In addition to neighbors on the current level, add the same node in the layer below
        # Make sure not already on bottom level
        if current_level != 0:
            # Make sure node wasn't already visited
            if (current_node, current_level-1) not in visited:
                    visited.add((current_node, current_level-1))
                    heapq.heappush(open_pq, (score, (current_node, current_level-1)))
                    
    pbar.close()     
    return node_scores

def traverseHNSW(hnsw_graphs, scoring_fn, num_to_search):
    
    (open_pq_start, 
    visited_start, 
    start_node_scores) = initialize_search(hnsw_graphs, scoring_fn)

    scores = traverse_queue(
                open_pq_start,
                visited_start,
                hnsw_graphs,
                num_to_search,
                start_node_scores,
                scoring_fn)

    return scores
