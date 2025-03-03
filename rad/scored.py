from abc import ABC, abstractmethod
import redis

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
    def __init__(self, redis_host='localhost', redis_port=6379, scored_name='scored'):
        self.r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.scored_list = f"{scored_name}:list"
        self.scored_set = f"{scored_name}:set"

    def getScore(self, key):
        score = self.r.hget(self.scored_set, str(key))
        return float(score) if score is not None else None
    
    def insert(self, key, score):
        self.r.hset(self.scored_set, str(key), str(score))
        self.r.rpush(self.scored_list, str(key))

    def save(self, path):
        with open(path, "w") as f:
            for key in self.r.lrange(self.scored_list, 0, -1):
                score = self.getScore(key)
                f.write(f"{key} {score}\n")

    def __len__(self):
        return self.r.llen(self.scored_list)