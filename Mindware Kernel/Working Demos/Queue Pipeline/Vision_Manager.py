import Queue_Manager as qm
from datetime import datetime, timezone
import time
import threading

class VisionThread(threading.Thread):

    def __init__(self, msg_queue):

        super().__init__(daemon=True)

        self.queue = msg_queue

        self.running = threading.Event()
        self.running.set()

    def stop(self):

        self.running.clear()

    def run(self):

        i = 0

        while self.running.is_set():

            msg = qm.TargetMessage(
                x=i,
                y=i,
                timestamp=datetime.now(timezone.utc)
            )

            print("[VISION] Broadcast:", msg)

            self.queue.enqueue(msg)

            i += 1
            time.sleep(0.5)
