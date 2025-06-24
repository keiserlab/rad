from abc import ABC, abstractmethod

class ScoredSet(ABC):
    @abstractmethod
    def getScore(self, key: int) -> float:
        pass

    @abstractmethod
    def insert(self, key: int, score: float):
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
        
        # Atomic insert script to prevent duplicates
        INSERT_SCRIPT = """
        local key = ARGV[1]
        local score = ARGV[2]
        local exists = redis.call('HEXISTS', KEYS[2], key)
        if exists == 0 then
            redis.call('HSET', KEYS[2], key, score)
            redis.call('RPUSH', KEYS[1], key)
        end
        """
        self.insert_script = self.r.register_script(INSERT_SCRIPT)

    def getScore(self, key):
        score = self.r.hget(self.scored_set, str(key))
        if score is not None:
            # Decode bytes to string if necessary
            if isinstance(score, bytes):
                score = score.decode('utf-8')
            return float(score)
        return None
    
    def insert(self, key, score):
        self.insert_script(keys=[self.scored_list, self.scored_set], args=[str(key), str(score)])

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
