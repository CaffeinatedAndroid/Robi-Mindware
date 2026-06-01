import Queue_Manager as queue
import threading
import time

class ServoThread(threading.Thread):

    def __init__(self, queue):
        super().__init__(daemon=True)

        self.queue = queue
        self.running = True

    def run(self):

        while self.running:

            msg = self.queue.dequeue()

            if msg is None:
                continue

            print("[SERVO] Consumed:", msg)


    def stop(self):
        self.running = False
