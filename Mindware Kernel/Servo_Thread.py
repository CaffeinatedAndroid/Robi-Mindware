import threading
import queue
import time
from datetime import datetime


class ServoThread(threading.Thread):

    def __init__(self, bus):

        super().__init__()

        self.queue = bus.subscribe(
            "target_detected" #TODO CHANGE FROM STRINGS
        )
        self.bus = bus
        self.SERVO_LIMITS = [0, 1500]


        self.running = True

    def stop(self):

        self.running = False

    def run(self):


        while self.running:

            envelope = self.queue.get()
            self.bus.publish("servo_command", [4,-300]) #NOTE RIGHT KNEE
            time.sleep(0.4)
            self.bus.publish("servo_command", [9,300]) #NOTE LEFT KNEE
            time.sleep(0.4)

            self.bus.publish("servo_command", [5,0]) #NOTE RIGHT ANKLE
            time.sleep(0.4)
            self.bus.publish("servo_command", [10,0]) #NOTE LEFT ANKLE
            time.sleep(0.4)

            self.bus.publish("servo_command", [3,0]) #NOTE RIGHT THIGH
            time.sleep(0.4)
            self.bus.publish("servo_command", [8,0]) #NOTE LEFT THIGH
            time.sleep(0.4)


            self.bus.publish("servo_command", [2,50]) #NOTE RIGHT HIP
            time.sleep(0.4)
            self.bus.publish("servo_command", [7,-50]) #NOTE LEFT HIP
            time.sleep(0.4)


            self.bus.publish("servo_command", [6,-50]) #NOTE RIGHT FOOT
            time.sleep(0.4)
            self.bus.publish("servo_command", [11,50]) #NOTE LEFT FOOT
            time.sleep(0.4)


            self.bus.publish("servo_command", [16,0]) #NOTE RIGHT SHOULDER
            time.sleep(0.4)
            self.bus.publish("servo_command", [19,0]) #NOTE LEFT SHOULDER
            time.sleep(0.4)

            self.bus.publish("servo_command", [17,500]) #NOTE RIGHT UPPER ARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [20,-500]) #NOTE LEFT UPPER ARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,300]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)



            self.bus.publish("servo_command", [16,800]) #NOTE RIGHT SHOULDER
            time.sleep(0.4)
            self.bus.publish("servo_command", [19,100]) #NOTE LEFT SHOULDER
            time.sleep(0.4)

            self.bus.publish("servo_command", [17,200]) #NOTE RIGHT UPPER ARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [20,-500]) #NOTE LEFT UPPER ARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,1200]) #NOTE RIGHT FOREARM
            time.sleep(1)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,300]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,1200]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,300]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,1200]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(0.4)

            self.bus.publish("servo_command", [18,300]) #NOTE RIGHT FOREARM
            time.sleep(0.4)
            self.bus.publish("servo_command", [21,-500]) #NOTE LEFT FOREARM
            time.sleep(2)




            if envelope is None:
                break

            # #NOTE STAND POSE
            # self.bus.publish("servo_command", "2:100") #NOTE RIGHT HIP
            # self.bus.publish("servo_command", "7:-100") #NOTE LEFT HIP
            #
            # self.bus.publish("servo_command", "3:0") #NOTE RIGHT THIGH
            # self.bus.publish("servo_command", "8:0") #NOTE LEFT THIGH
            #
            # self.bus.publish("servo_command", "4:-300") #NOTE RIGHT KNEE
            # self.bus.publish("servo_command", "9:300") #NOTE LEFT KNEE
            #
            # self.bus.publish("servo_command", "5:0") #NOTE RIGHT ANKLE
            # self.bus.publish("servo_command", "10:0") #NOTE LEFT ANKLE
            #
            # self.bus.publish("servo_command", "6:-50") #NOTE RIGHT FOOT
            # self.bus.publish("servo_command", "11:50") #NOTE LEFT FOOT



            msg = envelope.payload
            #command = 13, msg.x #TODO MAKE SURE TO ADD ID
            #self.bus.publish("servo_command", command)
            if envelope.topic == "target_detected":
                x = msg.x
                y = msg.y
                x_Mapped = max(self.SERVO_LIMITS[0], min(x, self.SERVO_LIMITS[1]))
                y_Mapped = max(self.SERVO_LIMITS[0], min(y, self.SERVO_LIMITS[1]))
                commandNeck = 13, -x_Mapped*2 + 400
                commandRoll = 14, -y_Mapped/3
                commandPitch = 15, y_Mapped/3

                # self.bus.publish("servo_command", commandNeck)
                # self.bus.publish("servo_command", commandRoll)
                # self.bus.publish("servo_command", commandPitch)
                #
                # self.bus.publish("servo_command", [2,50]) #NOTE RIGHT HIP
                # self.bus.publish("servo_command", [7,-50]) #NOTE LEFT HIP
                #
                # self.bus.publish("servo_command", [3,0]) #NOTE RIGHT THIGH
                # self.bus.publish("servo_command", [8,0]) #NOTE LEFT THIGH
                #
                # self.bus.publish("servo_command", [4,-300]) #NOTE RIGHT KNEE
                # self.bus.publish("servo_command", [9,300]) #NOTE LEFT KNEE
                #
                # self.bus.publish("servo_command", [5,0]) #NOTE RIGHT ANKLE
                # self.bus.publish("servo_command", [10,0]) #NOTE LEFT ANKLE
                #
                # self.bus.publish("servo_command", [6,-50]) #NOTE RIGHT FOOT
                # self.bus.publish("servo_command", [11,50]) #NOTE LEFT FOOT



            print(
                f"SERVO: "
                f"Topic= {envelope.topic}, "
                f"Data= {msg.x}, {msg.y}"
            )

            envelope.timestamp
            latency = (
            datetime.utcnow() -
            envelope.timestamp
            ).total_seconds()

            print(f"LATENCY: {latency * 1000:.3f} ms")

        print("SERVO: Stopped")


