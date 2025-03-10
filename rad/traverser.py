from .priority_queue import PriorityQueue, RedisPQ
from .scored import ScoredSet, RedisScoredSet
from .visited import VisitedSet, RedisVisited
from .hnsw_server import HNSWServer
from .redis_server import RedisServer

from typing import Union
import multiprocessing
import redis
import time

class RADTraverser:
    def __init__(self,
                 hnsw,
                 scoring_fn: callable,
                 priority_queue: Union[PriorityQueue, str] = 'redis',
                 visited_set: Union[VisitedSet, str] = 'redis',
                 scored_set: Union[ScoredSet, str] = 'redis',
                 **kwargs):
        
        # TODO: Probably restructure this a bit
        if any(x == "redis" for x in (priority_queue, visited_set, scored_set)):
            self._init_redis_client(**kwargs)

        self.priority_queue = self._init_pq(priority_queue, **kwargs)
        self.visited_set = self._init_visited(visited_set, **kwargs)
        self.scored_set = self._init_scored(scored_set, **kwargs)
        
        self.scoring_fn = scoring_fn

        self.hnsw = hnsw
        self.hnsw_server = HNSWServer(self.hnsw)

    def _init_redis_client(self, redis_host: str = None, redis_port: int = 6379, **kwargs) -> None:
        if redis_host is not None:
            print(f'Connecting to established redis server at {redis_host}:{redis_port}')
            self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port)
        else:
            print(f'Starting local redis server on port {redis_port}')
            self.redis_server = RedisServer(redis_port=redis_port, **kwargs)
            self.redis_client = self.redis_server.getClient()
   
    def _init_pq(self, priority_queue: Union[PriorityQueue, str], **kwargs) -> PriorityQueue:
        if isinstance(priority_queue, PriorityQueue):
            return priority_queue
    
        if priority_queue == "redis":
            return RedisPQ(redis_client=self.redis_client, **kwargs)
        else:
            raise ValueError("priority_queue must be 'redis' or a PriorityQueue instance")
        
    def _init_visited(self, visited_set: Union[VisitedSet, str], **kwargs) -> VisitedSet:
        if isinstance(visited_set, VisitedSet):
            return visited_set
    
        if visited_set == "redis":
            return RedisVisited(redis_client=self.redis_client, **kwargs)
        else:
            raise ValueError("visited_set must be 'redis' or a VisitedSet instance")

    def _init_scored(self, scored_set: Union[ScoredSet, str], **kwargs) -> ScoredSet:
        if isinstance(scored_set, ScoredSet):
            return scored_set
        
        if scored_set == "redis":
            return RedisScoredSet(redis_client=self.redis_client, **kwargs)
        else:
            raise ValueError("scored_set must be 'redis' or a ScoredSet instance")

    @staticmethod
    def _traverse(hnsw,
                  scoring_fn,
                  priority_queue,
                  visited_set,
                  scored_set,
                  timeout=None,
                  n_to_score=None,
                  **kwargs):

        if timeout is None and n_to_score is None:
            raise ValueError("Must provide a timeout or number of molecules to score")

        start_time = time.time()
        while True:
            if timeout is not None and time.time() - start_time >= timeout:
                print("Timeout reached")
                return

            best_mol = priority_queue.pop()
            if best_mol is None:
                print('Queue is empty')
                return

            cur_node_id, cur_level, cur_score = best_mol
            neighbors = hnsw.get_neighbors(cur_node_id, cur_level)
            
            for i in range(0, len(neighbors), 2):
                neighbor_id, neighbor_key = neighbors[i], neighbors[i+1]
                # If we've visited the neighbor already, continue
                if visited_set.checkAndInsert(node_id=neighbor_id, level=cur_level):
                    continue
                # Get the neighbor score if we have it
                score = scored_set.getScore(neighbor_key)
                # Otherwise calculate it
                if score is None:
                    score = scoring_fn(neighbor_key, **kwargs)
                    scored_set.insert(key=neighbor_key, score=score)
                    if n_to_score is not None and len(scored_set) > n_to_score:
                        print("Scored desired number of nodes")
                        return
                # Insert the neighbor into the queue
                priority_queue.insert(node_id=neighbor_id, level=cur_level, score=score)
            # Also add the current node a level down
            if (cur_level > 0 and not visited_set.checkAndInsert(node_id=cur_node_id, level=cur_level-1)):
                priority_queue.insert(node_id=cur_node_id, level=cur_level-1, score=cur_score)

    def traverse(self,
                 n_workers: int,
                 **kwargs):

        if n_workers == 1:
            self._traverse(self.hnsw_server, self.scoring_fn, self.priority_queue, self.visited_set, self.scored_set, **kwargs)
        else:
            processes = []
            for _ in range(n_workers):
                p = multiprocessing.Process(
                    target=self._traverse,
                    args=(self.hnsw_server, self.scoring_fn, self.priority_queue, self.visited_set, self.scored_set),
                    kwargs=kwargs)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    # TODO: multiprocess the beginning scoring
    def prime(self, **kwargs):
        top_level_nodes = self.hnsw.get_top_level_nodes()
        for i in range(0, len(top_level_nodes), 2):
            node_id, node_key = top_level_nodes[i], top_level_nodes[i+1]
            score = self.scoring_fn(node_key, **kwargs)
            self.scored_set.insert(key=node_key, score=score)
            self.visited_set.checkAndInsert(node_id=node_id, level = max(0, self.hnsw.max_level-1))
            self.priority_queue.insert(node_id=node_id, level=max(0,self.hnsw.max_level-1), score=score)


    # TODO: Probably a cleaner way to shutdown the redis server and hnsw server (at least for end user)
    def shutdown(self, **kwargs):
        self.hnsw_server.shutdown()
        if self.redis_server:
            self.redis_server.shutdown(**kwargs)