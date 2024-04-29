import hnswlib_rad
from tqdm import tqdm
import graph_tool.all as gt
import struct
import time

def int8_array_to_uint32(array):
    byte_data = bytes(array)
    label = struct.unpack('<I', byte_data)[0]
    return label

def int8_array_to_uint64(array):
    byte_data = bytes(array)
    label = struct.unpack('<Q', byte_data)[0]
    return label

def constructGraphs(hnsw_index, upper_level_neighbors):
    max_level = hnsw_index['max_level']
    layer_graphs = [gt.Graph() for i in range(0,max_level+1)]
    edges_to_add = [[] for i in range(0,max_level+1)]
    
    num_elements = hnsw_index['cur_element_count']
    size_data_per_element = hnsw_index['size_data_per_element']
    label_offset = hnsw_index['label_offset']
    offset_data = hnsw_index['offset_data']
    element_levels = hnsw_index['element_levels']


    for i in tqdm(range(num_elements), desc="Constructing graph_tool graphs"):
        element_data = hnsw_index['data_level0'][i*size_data_per_element:i*size_data_per_element+size_data_per_element]
        element_label = int8_array_to_uint64(element_data[label_offset:])
        level0_neighbor_data = element_data[:offset_data]
        level0_neighbors = getNeighborsFromNeighborArray(level0_neighbor_data)
        
        for neighbor in level0_neighbors:
            edges_to_add[0].append((element_label, neighbor))
        
        element_max_level = element_levels[element_label]
        for level in range(1,element_max_level+1):
            neighbor_data = upper_level_neighbors[element_label][level]
            neighbors = getNeighborsFromNeighborArray(neighbor_data)
            
            for neighbor in neighbors:
                edges_to_add[level].append((element_label, neighbor))
                

        if (i+1)%1_000_000 == 0 or (i+1)==num_elements:
            print(f"Adding batch {int(i/1000000)}/{int(num_elements/1000000)} to graphs")
            for level, graph in enumerate(layer_graphs):
                graph.add_edge_list(edges_to_add[level])
            edges_to_add = [[] for k in range(0,max_level+1)]

    # Sometimes the saved max level won't actually contain nodes. Not entirely sure why
    if layer_graphs[-1].num_vertices() == 0:
        return layer_graphs[:-1]

    return layer_graphs

def getNeighborsFromNeighborArray(neighbor_array):
    num_neighbors = int8_array_to_uint32(neighbor_array[0:4])
    neighbor_ids = set()
    for neighbor_int8 in [neighbor_array[i:i+4] for i in range(4,(num_neighbors+1)*4,4)]:
        neighbor_ids.add(int8_array_to_uint32(neighbor_int8))
    return neighbor_ids

def processLinkLists(hnsw_index):
    link_lists = hnsw_index['link_lists']
    element_levels = hnsw_index['element_levels']
    size_links_per_element = hnsw_index['size_links_per_element']
    results = {} # idx: {level: [neighbor array]}
    start_idx = 0
    for i, max_level in tqdm(enumerate(element_levels), desc="Formatting hnswlib Neighbor Data"):
        for level in range(1,max_level+1):
            if i in results:
                results[i][level] = link_lists[start_idx:start_idx+size_links_per_element]
            else:
                results[i] = {level: link_lists[start_idx:start_idx+size_links_per_element]}
            start_idx += size_links_per_element
        if i > hnsw_index['cur_element_count']:
            break
    return results

def getGraphs(data, ef_construction=400, M=16):
    p = hnswlib_rad.TanimotoIndex(dim=data.shape[1])

    p.init_index(max_elements=len(data), ef_construction=ef_construction, M=M)
    start = time.time()
    p.add_items(data)
    print(f"HNSW Construction time: {time.time()-start}")

    hnsw_index = p.get_ann_data()
    upper_level_neighbors = processLinkLists(hnsw_index)
    hnsw_graphs = constructGraphs(hnsw_index, upper_level_neighbors)
    return hnsw_graphs