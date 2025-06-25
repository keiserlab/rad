# RAD (Retrieval Augmented Docking)

RAD is a scalable virtual screening library using HNSW graphs and distributed computing. The architecture supports deployment from single machines to HPC clusters and cloud environments.


## Requirements

- Redis
- Python >=3.11

## Installation
```bash
git clone --recursive https://github.com/keiserlab/rad.git
cd rad
pip install . 
```

We also provide a Dockerfile containing all required software.

## Architecture Overview

RAD uses a service-oriented design with three main components:

1. **HNSW Service**: Handles HNSW neighbor searches and SMILES lookup
2. **Coordination Service**: Manages work distribution and state via Redis
3. **Distributed Workers**: Lightweight scoring processes that can run anywhere

## Running RAD

### Basic Workflow
1. Build HNSW graph from molecular fingerprints
2. Create SQLite database mapping node keys to SMILES
3. Define a SMILES-based scoring function
4. Initialize RAD services and run traversal

### Constructing the HNSW
Constructing the HNSW graph consists of setting the construction parameters *expansion_add* and *connectivity* and then adding each molecule by providing a numerical key and its fingerprint.

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


### Creating SQLite Database for SMILES mapping
RAD integrates with SQLite to provide SMILES directly to scoring functions:

```python
import sqlite3

# Create database mapping HNSW keys to SMILES
conn = sqlite3.connect('molecules.db')
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE nodes (
        node_key INTEGER PRIMARY KEY,
        smi TEXT NOT NULL
    )
""")

# Insert SMILES data
for key, smiles in zip(keys, smiles):
    cursor.execute("INSERT INTO nodes (node_key, smi) VALUES (?, ?)", (key, smiles))

cursor.execute("CREATE INDEX idx_nodes_node_id ON nodes(node_key)")
conn.commit()
conn.close()
```

### Defining a SMILES-Based Scoring Function
Scoring functions receive SMILES strings and return a score. Numerically smaller scores are considered better. Here is a mock example:

```python
def score_fn(smiles: str) -> float:
    score = calculate_docking_score(smiles)
    return score  # Lower scores are better
```

### Initializing RAD Services
With the HNSW index, SMILES database, and scoring function ready, initialize the RAD traverser:

```python
from rad.hnsw_service import create_local_hnsw_service
from rad.traverser import RADTraverser

# Create HNSW service with database integration
hnsw_service = create_local_hnsw_service(hnsw, database_path='molecules.db')

# Create traverser with SMILES-based scoring
traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=score_fn)
```

### Deployment Modes

**Local Deployment** (single machine):
```python
traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=score_fn)
```

**Distributed Deployment** (HPC/cloud):
```python
traverser = RADTraverser(
    hnsw_service=hnsw_service, 
    scoring_fn=score_fn,
    redis_host='head-node.cluster',
    redis_port=6379,
    namespace='job_12345'
)
```

**Remote HNSW Service**:
```python
from rad.hnsw_service import create_remote_hnsw_service

# Start HNSW server elsewhere
# python scripts/start_hnsw_server.py --database-path molecules.db --port 8000

hnsw_service = create_remote_hnsw_service("http://hnsw-server:8000")
traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=score_fn)
```

### Priming the RAD Traverser
The traverser is 'primed' by finding and scoring the nodes on the top layer of the HNSW graph and initializing the priority queue. This should only be run once.

```
traverser.prime()
```

### Performing the traversal
The traversal proceeds until a maximum number of molecules is scored or a timeout is reached:

```python
# Run traversal until 100k molecules are scored
traverser.traverse(n_workers=4, n_to_score=100_000)

# Or run traversal for a specific time
traverser.traverse(n_workers=4, timeout=3600)  # 1 hour
```

### Accessing the results
RAD provides two methods for accessing results:

**Traversal Order** (search path analysis):
```python
# Get molecules in the order they were discovered
molecules = traverser.get_molecules()  # All molecules
first_100 = traverser.get_molecules(100)  # First 100 molecules

for node_id, score, smiles in molecules:
    print(f"Node {node_id}: {smiles} (score: {score})")
```

**Best Molecules** (results analysis):
```python
# Get top-scoring molecules regardless of discovery order
best_molecules = traverser.get_best_molecules(10)  # Top 10 by score

for node_id, score, smiles in best_molecules:
    print(f"Top hit: {smiles} (score: {score})")
```


### Service Management and Cleanup
Gracefully shutdown all services:
```python
traverser.shutdown()
```

### Advanced Usage

**Starting HNSW Server Independently**:
```bash
# Start dedicated HNSW server with database
python scripts/start_hnsw_server.py \
  --hnsw-path /data/index.usearch \
  --database-path /data/molecules.db \
  --host 0.0.0.0 \
  --port 8000
```


## Example Usage

The `examples/` folder contains a Jupyter notebook demonstrating the construction and traversal of the DUDE-Z DOCK HNSW investigated in the [original RAD paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00683).

For a larger billion-scale application and integration with Chemprop see the repo at https://github.com/bwhall61/lsd


## Security Considerations

**Important**: The included HNSW server (`scripts/start_hnsw_server.py`) is designed for research and development use. Before deploying to production or public networks, implemenent additional security features.

The default configuration binds to `127.0.0.1` (localhost only) for security. Only bind to public interfaces (`0.0.0.0`) in trusted network environments.

## References
The original [HNSW paper](https://arxiv.org/abs/1603.09320) by Yury Malkov and Dmitry Yashunin.

The original [RAD paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00683) by Brendan Hall and Michael Keiser.

The [lsd.docking.org paper](https://www.biorxiv.org/content/10.1101/2025.02.25.639879v1) by Brendan Hall, Tia Tummino, et al. shows a billion-scale application and integration with Chemprop ML models.

And then most importantly, the HNSW graph code is built on the [usearch](https://github.com/unum-cloud/usearch) library so large thanks to Ash Vardanian for his awesome HNSW library!
