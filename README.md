# RAD (Retrieval Augmented Docking)

## Installation
```
git clone https://github.com/keiserlab/rad.git
cd rad
conda env create -f environment.yml
conda activate rad
pip install . 
```
Please note that Python >=3.7 is required.

## Running RAD
There are two primary things that must be done to run RAD: 
- Constructing the HNSW
- Defining a scoring function to use for traversal

### Constructing the HNSW
Constructing the HNSW graphs is as simple as passing in your fingerprints and the HNSW construction parameters *ef_construction* and *M*.
*ef_construction* controls the number of candidates considered as potential neighbors during element insertion, while *M* controls how many of these candidates are actually connected to the inserted element.

```
from rad.construction import getGraphs
data = ...
hnsw_graphs = getGraphs(data, ef_construction=400, M=16)
```

Note that the data is expected to be an (*n* x *d*) numpy array where each row is a fingerprint. Each fingerprint will be assigned a node_id according to its position in this list.

While not strictly necessary, construction is much faster if you pack bit fingerprints into uint8 arrays. E.g., turning a 1024-bit fingerprint into a 128 uint8 fingerprint with `np.packbits ()`.

### Defining a scoring function and traversing the HNSW
To traverse the HNSW graphs, you must provide a scoring function that maps a fingerprint's node_id (assigned during construction) to a score, where numerically smaller scores are better. This scoring function is then passed to the traversal function, along with the HNSW graphs, and the number of molecules to score before stopping traversal.

```
from rad.traversal import traverseHNSW

# Example of getting the fingerprint and using it to calculate the score
def score_fn(node_id):
    fp = data[node_id]
    score = score_from_fp(fp)
    return score

traversed_nodes = traverseHNSW(hnsw_graphs, score_fn, num_to_search=1000)
```

This will return a dictionary of node_ids and their corresponding scores. The order in which the node_ids appear in the dictionary is the order in which they were traversed (as long as Python >=3.7 is used)

### Example
In the example folder, there is a jupyter notebook for constructing and traversing the DUDE-Z DOCK HNSW investigated in the paper.

## References
[The original HNSW paper](https://github.com/nmslib/hnswlib) by Yury Malkov and Dmitry Yashunin

Most of this code is built on the [hnswlib](https://github.com/nmslib/hnswlib) library and the [PR](https://github.com/nmslib/hnswlib/pull/364) by [@psobot](https://github.com/psobot) which implemented python bindings for indices using integer data types
