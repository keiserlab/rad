# RAD (Retrieval Augmented Docking)

## Requirements
Redis

Python >=3.11

## Installation
```
git clone --recursive https://github.com/keiserlab/rad.git
cd rad
pip install . 
```

We also provide a Dockerfile containing all required software.

## Running RAD
There are two primary things to run RAD: 
- Constructing the HNSW graph
- Defining a scoring function to use for traversal

### Constructing the HNSW
Constructing the HNSW graph consists of setting the construction parameters *expansion_add* and *connectivity* and then adding each molecule by providing a numerical key and a its fingerprint.

*expansion_add* controls the number of candidates considered as potential neighbors during element insertion, while *connectivity* controls how many of these candidates are actually connected to the inserted element.

```
from usearch.index import Index

hnsw = Index(
    ndim = 1024, # 1024 bit fingerprint
    dtype='b1', # For packed binary fingerprints
    metric='tanimoto',
    connectivity = 8,
    expansion_add = 400
)

fingerprints = ...
keys = np.arange(len(fingerprints))

hnsw.add(keys, fingerprints, log="Building HNSW")
```

The fingerprints are expected to be an (*n* x *d/8*) numpy array where each row is a packed binary fingerprint. e.g turning a 1024-bit binary fingerprint into a 128 uint8 fingerprint with `np.packbits()`. See the example notebook for more details.


### Defining a scoring function
To traverse the HNSW graphs, you must provide a scoring function that maps a molecule's key to a score, where numerically smaller scores are better. Here is a mock example of a scoring function:

```
def score_fn(key):
    fp = fingerprints[key]
    score = score_from_fp(fp)
    return score
```

### Initializing a RAD Traverser
With the HNSW graphs built and the scoring function defined, we can initialize a `RADTraverser`

```
from rad.traverser import RADTraverser

traverser = RADTraverser(hnsw=hnsw, scoring_fn=score_fn)
```

This initialization does two things:
1. Starts a process which handles the HNSW neighbor queries.
2. Starts a process which runs a redis server and handles the traversal queue.

You can also connect to an already running redis server by passing the host and port of the server. This is useful for HPC environments where a single node can manage the traversal queue for many worker nodes doing the scoring:

```
traverser = RADTraverser(hnsw=hnsw, scoring_fn=score_fn, redis_host='xxx:xxx:xxx', redis_port=6379)
```

### Priming the RAD Traverser
The traverser is 'primed' by finding and scoring the nodes on the top layer of the HNSW graph and initializing the priority queue. This should only be run once.

```
traverser.prime()
```

### Performing the traversal
The traversal then proceeds until a max number of molecules is scored or a timeout is reached. 

```
# Note that n_workers must be set to 1 for now until I fix a bug.
traverser.traverse(n_workers=1, n_to_score=100,000) # Run the traversal until 100k molecules are scored
```
or 
```
traverser.traveser(n_workers=1, timeout=120) # Run the traversal for 2 minutes
```

### Accessing the results
The results can be accessed by looping over the scored set which contains the keys in the order that they were traversed. 

```
results = []
for key, score in traverser.scored_set:
    results.append((key,score))
```

### Shutting down the HNSW and redis servers
To gracefully shut down the HNSW and redis servers:
```
traverser.shutdown()
```

### Example
In the example folder, there is a jupyter notebook for constructing and traversing the DUDE-Z DOCK HNSW investigated in the [original RAD paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00683).

For a larger billion-scale application and integration with Chemprop see the repo at https://github.com/bwhall61/lsd

## References
The original [HNSW paper](https://arxiv.org/abs/1603.09320) by Yury Malkov and Dmitry Yashunin.

The original [RAD paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00683) by Brendan Hall and Michael Keiser.

The [lsd.docking.org paper](https://www.biorxiv.org/content/10.1101/2025.02.25.639879v1) by Brendan Hall, Tia Tummino, et al. shows a billion-scale application and integration with Chemprop ML models.

And then most importantly, the HNSW graph code is built on the [usearch](https://github.com/unum-cloud/usearch) library so large thanks to Ash Vardanian for his awesome HNSW library!
