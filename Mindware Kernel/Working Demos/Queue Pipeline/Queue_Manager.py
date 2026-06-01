import threading
from dataclasses import dataclass
from datetime import datetime, timezone
import time
import Vision_Manager
import Servo_Manager


#NOTE: OBJECTS TO QUEUE
@dataclass
class TargetMessage:
    x: int
    y: int
    timestamp: datetime

    def __repr__(self):
        ts_str = self.timestamp.strftime("%d-%m-%y %H:%M:%S %Z")
        return f"TargetMessage(X={self.x}, Y={self.y}, Time:'{ts_str}')"

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None



#NOTE: QUEUE LOGIC
class Queue:

    def __init__(self):

        #NOTE SETUP QUEUE
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.front = None
        self.rear = None
        self.size = 0    #NOTE: INITIAL QUEUE SIZE
        self.shutdown = False


    def __len__(self):
        with self.condition:
            return self.size

    def __repr__(self):
        with self.condition:
            items = []
            current = self.front
            while current:
                items.append(str(current.value))
                current = current.next

            return ", ".join(items)

    def enqueue(self, value):

        with self.condition:

            new_node = Node(value)

            if self.rear is None:
                self.front = self.rear = new_node
            else:
                self.rear.next = new_node
                self.rear = new_node

            self.size += 1

            self.condition.notify_all()


    def dequeue(self):

        with self.condition:

            while self.front is None and not self.shutdown:
                self.condition.wait()

            if self.shutdown and self.front is None:
                return None

            dequeue_value = self.front.value
            self.front = self.front.next

            if self.front is None:
                self.rear = None

            self.size -= 1

            return dequeue_value



    def peek(self):
        with self.condition:
            if self.front is None:
                raise IndexError("NOTICE: Queue is empty!")
            return self.front.value

    def is_empty(self):
        with self.condition:
            return self.front is None


    def close(self):
        with self.condition:
            self.shutdown = True
            self.condition.notify_all()

if __name__ == '__main__':
    queue = Queue()

    vision = Vision_Manager.VisionThread(queue)
    servo = Servo_Manager.ServoThread(queue)
    #
    print("QUEUE: Initiating threads")
    vision.start()
    servo.start()
    print("QUEUE: Threads initiated")
    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt: #NOTE crtl c

        print("QUEUE: Shutting down threads")

        vision.stop()
        servo.stop()

        queue.close()

        vision.join()
        servo.join()
