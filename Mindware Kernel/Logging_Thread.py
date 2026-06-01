import threading
import queue


class LoggerThread(threading.Thread):

    def __init__(self, bus):

        super().__init__()

        self.queue = bus.subscribe(
            "target_detected"
        )

        self.running = True

    def stop(self):

        self.running = False

    def run(self):

        while self.running:

            try:
                envelope = self.queue.get(timeout=0.5)


            except queue.Empty:
                continue

            if envelope is None:
                break

            msg = envelope.payload

            print(f"LOGGER: Logged {msg}")

        print("LOGGER: Stopped")
