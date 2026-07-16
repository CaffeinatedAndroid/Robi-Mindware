import threading
import queue
import time
from datetime import datetime
from collections import defaultdict
import Message_Structure as messasgeStructure



class MessageBus:

    def __init__(self):

        self.lock = threading.Lock()

        # topic -> list of subscriber queues
        self.subscribers = defaultdict(list)

        self.shutdown = False

    def subscribe(self, topic, maxsize=10):

        q = queue.Queue(maxsize=maxsize)


        with self.lock:
            self.subscribers[topic].append(q)

        print(f"BUS: Subscriber added to '{topic}'")

        return q

    def publish(self, topic, payload):

        envelope = messasgeStructure.Envelope(
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

