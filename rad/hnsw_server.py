import multiprocessing

# TODO: Allow a single central HNSW server that can feed neighbors to jobs across an HPC
class HNSWServer:
    def __init__(self, hnsw):
        self.hnsw = hnsw
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._server_process, daemon=True)
        self.process.start()

    def _server_process(self):
        while True:
            request = self.request_queue.get()
            if request == "STOP":
                break
            node_id, level = request
            response = self.hnsw.get_neighbors(node_id, level)
            self.response_queue.put(response)

    def get_neighbors(self, node_id, level):
        self.request_queue.put((node_id, level))
        return self.response_queue.get()

    def shutdown(self):
        self.request_queue.put("STOP")
        self.process.join()