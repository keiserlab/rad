from abc import ABC, abstractmethod
from typing import Tuple
import redis

class PriorityQueue(ABC):
    @abstractmethod
    def pop(self) -> Tuple[int, int, float]:
        pass

    @abstractmethod
    def insert(self, node_id: int, level: int, score: float):
        pass

class RedisPQ(PriorityQueue):
    def __init__(self, redis_host='localhost', redis_port=6379, queue_name='pq'):
        self.r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_name = queue_name

        # Need popping to be atomic so multiple processes don't pop the same node
        POP_SCRIPT = """
        local result = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
        if #result == 0 then return nil end
        redis.call('ZREM', KEYS[1], result[1])
        return {result[1], result[2]}
        """
        self.pop_script = self.r.register_script(POP_SCRIPT)

    def pop(self):
        result = self.pop_script(keys=[self.queue_name])
        if not result:
            return None
        composite_key, score = result
        node_id, level = map(int, composite_key.split(":"))
        return node_id, level, float(score)
    
    def insert(self, node_id, level, score):
        self.r.zadd(self.queue_name, {f"{node_id}:{level}": score})