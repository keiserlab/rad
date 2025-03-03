from abc import ABC, abstractmethod
import redis

class VisitedSet(ABC):
    @abstractmethod
    def checkAndInsert(self, node_id: int) -> bool:
        pass

class RedisVisited(VisitedSet):
    def __init__(self, redis_host='localhost', redis_port=6379, visited_name='visited'):
        self.r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.visited_name = visited_name

        # Checking and inserting into the visited set needs to be atomic
        CHECK_SCRIPT = """
        if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1 then
            return 1
        else
            redis.call('SADD', KEYS[1], ARGV[1])
            return 0
        end
        """
        self.check_script = self.r.register_script(CHECK_SCRIPT)

    def checkAndInsert(self, node_id, level):
        composite_key = f"{node_id}:{level}"
        return bool(self.check_script(keys=[self.visited_name], args=[composite_key]))