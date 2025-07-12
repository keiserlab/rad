from abc import ABC, abstractmethod

class ScoredSet(ABC):
    @abstractmethod
    def getScore(self, node_id: int) -> float:
        pass

    @abstractmethod
    def insert(self, node_id: int, score: float, smiles: str = ""):
        pass

    @abstractmethod
    def get_molecules(self, n: int = None):
        """Get molecules with scores and SMILES in traversal/insertion order."""
        pass
    
    @abstractmethod
    def get_best_molecules(self, n: int = None):
        """Get molecules with scores and SMILES sorted by best scores."""
        pass

    @abstractmethod
    def __len__(self):
        pass

class RedisScoredSet(ScoredSet):
    def __init__(self, redis_client=None, scored_name='scored', **kwargs):
        if redis_client is None:
            raise ValueError("RedisScoredSet requires a valid Redis client instance.")

        self.r = redis_client
        self.scored_list = f"{scored_name}:list"
        self.scored_set = f"{scored_name}:set"
        self.smiles_set = f"{scored_name}:smiles"  # Store SMILES separately
        
        # Atomic insert script to prevent duplicates and store SMILES
        INSERT_SCRIPT = """
        local node_id = ARGV[1]
        local score = ARGV[2]
        local smiles = ARGV[3]
        local exists = redis.call('HEXISTS', KEYS[2], node_id)
        if exists == 0 then
            redis.call('HSET', KEYS[2], node_id, score)
            redis.call('HSET', KEYS[3], node_id, smiles)
            redis.call('RPUSH', KEYS[1], node_id)
        end
        """
        self.insert_script = self.r.register_script(INSERT_SCRIPT)

    def getScore(self, node_id):
        score = self.r.hget(self.scored_set, str(node_id))
        if score is not None:
            # Decode bytes to string if necessary
            if isinstance(score, bytes):
                score = score.decode('utf-8')
            return float(score)
        return None
    
    def insert(self, node_id, score, smiles=""):
        self.insert_script(keys=[self.scored_list, self.scored_set, self.smiles_set], 
                          args=[str(node_id), str(score), str(smiles)])
    
    def get_molecules(self, n=None):
        """Get molecules with scores and SMILES in traversal/insertion order."""
        # Get all node_ids from the list (in insertion order)
        node_ids = self.r.lrange(self.scored_list, 0, -1)
        
        if n is not None:
            node_ids = node_ids[:n]
        
        molecules = []
        for node_id in node_ids:
            # Decode bytes to string if necessary
            if isinstance(node_id, bytes):
                node_id = node_id.decode('utf-8')
            
            score = self.getScore(node_id)
            smiles = self.r.hget(self.smiles_set, node_id)
            
            if smiles and isinstance(smiles, bytes):
                smiles = smiles.decode('utf-8')
            
            molecules.append((int(node_id), score, smiles or ""))
        
        return molecules
    
    def get_best_molecules(self, n=None):
        """Get molecules with scores and SMILES sorted by best scores (lowest scores first)."""
        # Get all molecules first
        all_molecules = self.get_molecules()
        
        # Sort by score (lower is better for docking scores)
        sorted_molecules = sorted(all_molecules, key=lambda x: x[1])
        
        if n is not None:
            return sorted_molecules[:n]
        
        return sorted_molecules

    def save(self, path):
        with open(path, "w") as f:
            for key, score in self:
                f.write(f"{key} {score}\n")

    def __iter__(self):
        for key in self.r.lrange(self.scored_list, 0, -1):
            # Decode bytes to string if necessary
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            yield (int(key), self.getScore(key))

    def __len__(self):
        return self.r.llen(self.scored_list)
