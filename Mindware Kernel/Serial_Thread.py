import threading
import queue
import time
import serial
from datetime import datetime


class SerialThread(threading.Thread):

    def __init__(self, bus):

        super().__init__()

        self.queue = bus.subscribe("servo_command")

        self.running = True

        self.COMMS_PORT = 'COM4'
        self.BAUDRATE = 115200
        #self.SERVO_LIMITS = [0, 1500]

        self.arduino = None

        self.initialize_communications()

    def initialize_communications(self):

        try:
            self.arduino = serial.Serial(
                self.COMMS_PORT,
                self.BAUDRATE,
                timeout=1
            )

            time.sleep(2)

            print("Communications Established")
            print(f"Port: {self.COMMS_PORT}")
            print(f"Speed: {self.BAUDRATE}")

        except Exception as error:

            print("Failed to initialize communications")
            print(f"Error: {error}")

            self.arduino = None

    def stop(self):

        self.running = False

        if self.arduino and self.arduino.is_open:
            self.arduino.close()

    def run(self):

        while self.running:

            try:
                envelope = self.queue.get(timeout=0.1)

            except queue.Empty:
                continue

            if envelope is None:
                break

            if not self.arduino:
                continue

            command = envelope.payload

            servo_id = command[0]

            angle = command[1]


            message = f"{servo_id}:{angle}\n"

            try:

                self.arduino.write(message.encode())

                print(
                    f"SERIAL: "
                    f"Topic={envelope.topic}, "
                    f"Data={message.strip()}"
                )

                latency = (
                    datetime.utcnow() -
                    envelope.timestamp
                ).total_seconds()

                print(f"LATENCY: {latency * 1000:.3f} ms")

            except serial.SerialException as error:

                print(f"Serial write failed: {error}")

            time.sleep(0.01)

        print("SERIAL: Stopped")


#
# COMMS_PORT = 'COM4'
# BAUDRATE = 115200
#
# SERVO_LIMITS = [0, 1500]
#
# arduino = None
#
#
# def InitializeCommunications():
#     global arduino
#
#     try:
#         # Open serial connection
#         arduino = serial.Serial(COMMS_PORT, BAUDRATE, timeout=10)
#
#         # Wait for Arduino reset
#         time.sleep(2)
#
#         print(f"Communications Established")
#         print(f"Port: {COMMS_PORT}")
#         print(f"Speed: {BAUDRATE}")
#
#         # Optional startup move
#         #MoveServo(13, 700)
#
#     except Exception as e:
#         print(f"Failed to initialize communications")
#         print(f"Error: {e}")
#
#
# def MoveServo(servo_ID, servo_Angle):
#
#     if arduino is None:
#         print("Serial connection not initialized.")
#         return
#
#     if isinstance(servo_Angle, int) and SERVO_LIMITS[0] <= servo_Angle <= SERVO_LIMITS[1]:
#
#         command = f"{servo_ID}:{servo_Angle}\n"
#
#         arduino.write(command.encode())
#
#         print(f"Sent: {command.strip()}")
#
#     else:
#         print("Invalid servo angle.")
#
#
# def DetachServo(servo_ID):
#
#     if arduino is None:
#         return
#
#     command = f"DETACH({servo_ID})\n"
#
#     arduino.write(command.encode())
#
#     print(f"Sent: {command.strip()}")
#
#
# def AttachServo(servo_ID):
#
#     if arduino is None:
#         return
#
#     command = f"ATTACH({servo_ID})\n"
#
#     arduino.write(command.encode())
#
#     print(f"Sent: {command.strip()}")
