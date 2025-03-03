from priority_queue import PriorityQueue, RedisPQ
from scored import ScoredSet, RedisScoredSet
from visited import VisitedSet, RedisVisited
from hnsw_server import HNSWServer

from typing import Union
import multiprocessing

class RADTraverser:
    def __init__(self,
                 hnsw,
                 scoring_fn: callable,
                 priority_queue: Union[PriorityQueue, str] = 'redis',
                 visited_set: Union[VisitedSet, str] = 'redis',
                 scored_set: Union[ScoredSet, str] = 'redis',
                 **kwargs):
        
        self.priority_queue = self._init_pq(priority_queue, **kwargs)
        self.visited_set = self._init_visited(visited_set, **kwargs)
        self.scored_set = self._init_scored(scored_set, **kwargs)
        
        self.scoring_fn = scoring_fn

        self.hnsw = hnsw
        self.hnsw_server = HNSWServer(self.hnsw)


    def _init_pq(self, priority_queue: Union[PriorityQueue, str], **kwargs) -> PriorityQueue:
        if isinstance(priority_queue, PriorityQueue):
            return priority_queue
    
        if priority_queue == "redis":
            return RedisPQ(**kwargs)
        else:
            raise ValueError("priority_queue must be 'redis' or a PriorityQueue instance")
        
    def _init_visited(self, visited_set: Union[VisitedSet, str], **kwargs) -> VisitedSet:
        if isinstance(visited_set, VisitedSet):
            return visited_set
    
        if visited_set == "redis":
            return RedisVisited(**kwargs)
        else:
            raise ValueError("visited_set must be 'redis' or a VisitedSet instance")

    def _init_scored(self, scored_set: Union[ScoredSet, str], **kwargs) -> ScoredSet:
        if isinstance(scored_set, ScoredSet):
            return scored_set
        
        if scored_set == "redis":
            return RedisScoredSet(**kwargs)
        else:
            raise ValueError("scored_set must be 'redis' or a ScoredSet instance")


    @staticmethod
    def _traverse(hnsw,
                  scoring_fn,
                  priority_queue,
                  visited_set,
                  scored_set,
                  timeout,
                  n_to_score = None,
                  **kwargs):

        start_time = time.time()
        while time.time() - start_time < timeout:
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
                    if len(scored_set) % 10000 == 0:
                        print(len(scored_set))
                    if n_to_score is not None and len(scored_set) > n_to_score:
                        print("Scored desired number of nodes")
                        return
                # Insert the neighbor into the queue
                priority_queue.insert(node_id=neighbor_id, level=cur_level, score=score)
            # Also add the current node a level down
            if (cur_level > 0 and not visited_set.checkAndInsert(node_id=cur_node_id, level=cur_level-1)):
                priority_queue.insert(node_id=cur_node_id, level=cur_level-1, score=cur_score)

    def traverse(self,
                 timeout: int,
                 n_workers: int,
                 **kwargs):

        processes = []
        for _ in range(n_workers):
            p = multiprocessing.Process(
                target=self._traverse,
                args=(self.hnsw_server, self.scoring_fn, self.priority_queue, self.visited_set, self.scored_set, timeout),
                kwargs=kwargs)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.hnsw_server.shutdown()
        

    # TODO: multiprocess the beginning scoring
    def primeGraph(self, **kwargs):
        top_level_nodes = self.hnsw.get_top_level_nodes()
        for i in range(0, len(top_level_nodes), 2):
            node_id, node_key = top_level_nodes[i], top_level_nodes[i+1]
            score = self.scoring_fn(node_key, **kwargs)
            self.scored_set.insert(key=node_key, score=score)
            self.visited_set.checkAndInsert(node_id=node_id, level = max(0, self.hnsw.max_level-1))
            self.priority_queue.insert(node_id=node_id, level=max(0,self.hnsw.max_level-1), score=score)