import time
import Vision_Thread as visionThread
import Servo_Thread as servoThread
import Logging_Thread as logThread
import Serial_Thread as serialThread
import Message_Bus as messageBus
import Simulation_Thread as simulationThread
import UI

#TODO ANNOTATIONS
#TODO PANDA 3D THREAD
#TODO IMU THREAD
#TODO MIC THREAD
#TODO LOGIC THREADS (VOICE RECOGNITION, CONVERSATIONS, INFERENCE)

bus = None
vision = None
servo = None
logger = None
serial = None

def start():

    print("APPLICATION: Starting threads")
    vision.start()
    servo.start()
    logger.start()
    serial.start()

    #NOTE START THE UI AND QUERY BUS, RUNS IN MAIN LOOP
    ui.run()

def shutdown():

    print("\nAPPLICATION: Shutting down")

    vision.stop()
    servo.stop()
    logger.stop()
    serial.stop()

    bus.close()

    vision.join()
    servo.join()
    logger.join()
    serial.join()
    ui.quit()

    print("APPLICATION: Shutdown complete")


if __name__ == "__main__":

    bus = messageBus.MessageBus()
    vision = visionThread.VisionThread(bus)
    servo = servoThread.ServoThread(bus)
    logger = logThread.LoggerThread(bus)
    serial = serialThread.SerialThread(bus)
    ui = UI.UIManager(bus)

    start()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        shutdown()
        print("APPLICATION: Exited program")
