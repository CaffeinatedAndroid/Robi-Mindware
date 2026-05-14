import serial
import time

# CONFIG
COMMS_PORT = 'COM4'
BAUDRATE = 115200

SERVO_LIMITS = [0, 1500]

arduino = None


def InitializeCommunications():
    global arduino

    try:
        # Open serial connection
        arduino = serial.Serial(COMMS_PORT, BAUDRATE, timeout=10)

        # Wait for Arduino reset
        time.sleep(2)

        print(f"Communications Established")
        print(f"Port: {COMMS_PORT}")
        print(f"Speed: {BAUDRATE}")

        # Optional startup move
        #MoveServo(13, 700)

    except Exception as e:
        print(f"Failed to initialize communications")
        print(f"Error: {e}")


def MoveServo(servo_ID, servo_Angle):

    if arduino is None:
        print("Serial connection not initialized.")
        return

    if isinstance(servo_Angle, int) and SERVO_LIMITS[0] <= servo_Angle <= SERVO_LIMITS[1]:

        command = f"{servo_ID}:{servo_Angle}\n"

        arduino.write(command.encode())

        print(f"Sent: {command.strip()}")

    else:
        print("Invalid servo angle.")


def DetachServo(servo_ID):

    if arduino is None:
        return

    command = f"DETACH({servo_ID})\n"

    arduino.write(command.encode())

    print(f"Sent: {command.strip()}")


def AttachServo(servo_ID):

    if arduino is None:
        return

    command = f"ATTACH({servo_ID})\n"

    arduino.write(command.encode())

    print(f"Sent: {command.strip()}")


# -------------------
# MAIN
# -------------------

InitializeCommunications()

MoveServo(13, 600)

#time.sleep(1)

#MoveServo(13, 900)

#time.sleep(5)

#DetachServo(13)
