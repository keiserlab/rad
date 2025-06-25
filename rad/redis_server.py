import multiprocessing
import subprocess
import time
import redis
from typing import Optional

class RedisServer:
    def __init__(self, redis_path="/usr/bin/redis-server", redis_port=6379, password: Optional[str] = None, **kwargs):
        self.redis_path = redis_path
        self.redis_port = redis_port
        self.password = password

        self.process = multiprocessing.Process(target=self._start_redis, daemon=True)
        self.process.start()

        # Create Redis client with optional password
        connection_params = {
            'host': 'localhost',
            'port': self.redis_port,
            'decode_responses': False
        }
        if self.password is not None:
            connection_params['password'] = self.password

        self.client = redis.StrictRedis(**connection_params)

        self._wait_for_startup(**kwargs)

    def _start_redis(self):
        redis_cmd = [self.redis_path, "--port", str(self.redis_port)]
        
        # Add password authentication if specified
        if self.password is not None:
            redis_cmd.extend(["--requirepass", self.password])
        
        self.redis_process = subprocess.Popen(redis_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.redis_process.wait()

    def _wait_for_startup(self, redis_connect_timeout=60):
        for _ in range(redis_connect_timeout):
            try:
                if self.client.ping():
                    return
            except redis.ConnectionError:
                print("Redis server not start yet - waiting 1 second")
                time.sleep(1)
        raise RuntimeError("Failed to start Redis server.")

    def getClient(self):
        return self.client

    def shutdown(self, save=False):
        self.client.shutdown(save=save)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()