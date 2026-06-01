import threading
import queue
import time

from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class Envelope:

    topic: str

    timestamp: datetime

    payload: object



# =========================
# MESSAGE TYPES
# =========================

@dataclass
class TargetMessage:
    x: int
    y: int
    timestamp: datetime

    def __repr__(self):

        ts = self.timestamp.strftime("%H:%M:%S")

        return (
            f"TargetMessage("
            f"x={self.x}, "
            f"y={self.y}, "
            f"time={ts})"
        )


# =========================
# MESSAGE BUS
# =========================

class MessageBus:

    def __init__(self):

        self.lock = threading.Lock()

        # topic -> list of subscriber queues
        self.subscribers = defaultdict(list)

        self.shutdown = False

    def subscribe(self, topic):

        q = queue.Queue(maxsize=10)


        with self.lock:
            self.subscribers[topic].append(q)

        print(f"BUS: Subscriber added to '{topic}'")

        return q

    def publish(self, topic, payload):

        envelope = Envelope(
            topic=topic,
            timestamp=datetime.utcnow(),
            payload=payload
        )

        with self.lock:

            if self.shutdown:
                return

            subscribers = list(self.subscribers[topic])

        for q in subscribers:

            try:

                q.put_nowait(envelope)

            except queue.Full:

                q.get_nowait()

                q.put_nowait(envelope)


    def close(self):

        with self.lock:

            self.shutdown = True

            # wake all subscribers
            for topic_queues in self.subscribers.values():

                for q in topic_queues:
                    q.put(None)

        print("BUS: Shutdown")


# =========================
# VISION THREAD
# =========================

class VisionThread(threading.Thread):

    def __init__(self, bus):

        super().__init__()

        self.bus = bus

        self.running = True

    def stop(self):

        self.running = False

    def run(self):

        x = 0

        while self.running:

            msg = TargetMessage(
                x=x,
                y=x * 2,
                timestamp=datetime.utcnow()
            )

            print(f"VISION: Publishing {msg}")

            self.bus.publish(
                "target_detected",
                msg
            )

            x += 1

            time.sleep(1)

        print("VISION: Stopped")


# =========================
# SERVO THREAD
# =========================

class ServoThread(threading.Thread):

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

            envelope = self.queue.get()

            if envelope is None:
                break

            msg = envelope.payload

            print(
                f"SERVO: "
                f"Topic={envelope.topic}, "
                f"Target=({msg.x}, {msg.y})"
            )

            envelope.timestamp
            latency = (
            datetime.utcnow() -
            envelope.timestamp
            ).total_seconds()

            print(f"LATENCY: {latency * 1000:.3f} ms")

        print("SERVO: Stopped")


# =========================
# LOGGER THREAD
# =========================

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


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    bus = MessageBus()

    vision = VisionThread(bus)

    servo = ServoThread(bus)

    logger = LoggerThread(bus)

    print("MAIN: Starting threads")

    vision.start()
    servo.start()
    logger.start()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print("\nMAIN: Shutting down")

        vision.stop()
        servo.stop()
        logger.stop()

        bus.close()

        vision.join()
        servo.join()
        logger.join()

        print("MAIN: Exit complete")
