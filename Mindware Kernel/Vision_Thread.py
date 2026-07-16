import threading
import time
from datetime import datetime

import cv2
import numpy as np

import Topics as topics


class VisionThread(threading.Thread):

    def __init__(self, bus):

        super().__init__()

        self.bus = bus

        self.running = True

        self.cx = None
        self.cy = None

        self.vid = cv2.VideoCapture(1)

    def stop(self):

        self.running = False

    def run(self):

        if not self.vid.isOpened():

            print("VISION: Failed to open camera")
            return

        while self.running:

            ret, frame = self.vid.read()

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Blur reduces noise
            gray = cv2.GaussianBlur(
                gray,
                (9, 9),
                2
            )

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                1.2,
                100,
                param1=100,
                param2=40,
                minRadius=75,
                maxRadius=100
            )

            if circles is not None:

                circles = np.round(circles[0, :]).astype("int")

                # Choose largest circle
                largest = max(
                    circles,
                    key=lambda c: c[2]
                )

                x, y, radius = largest

                self.cx = x
                self.cy = y

                # Draw circle
                cv2.circle(
                    frame,
                    (x, y),
                    radius,
                    (0, 255, 0),
                    2
                )

                # Draw center
                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 0, 255),
                    -1
                )

                target_msg = topics.TargetMessage(
                    x=self.cx,
                    y=self.cy,
                    timestamp=datetime.utcnow()
                )

                self.bus.publish(
                    "target_detected",
                    target_msg
                )

            frame_msg = topics.VisionFrameMessage(
                frame=frame,
                timestamp=datetime.utcnow()
            )

            self.bus.publish(
                "vision_frame",
                frame_msg
            )

            time.sleep(0.03)

        self.vid.release()

        print("VISION: Stopped")

#
# class VideoFeed(): #NOTE OPEN CV MANAGER
#
#     def __init__(self, parent,app, video_source=1, delay=30, coord_var= None):
#         self.app = app
#         self.parent = parent
#         self.delay = delay
#         self.cx = None
#         self.cy = None
#
#         self.previousCircle = (0, 0)
#         self.dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2
#
#         self.vid = cv2.VideoCapture(video_source)
#         self.coord_var = coord_var
#         ret, frame = self.vid.read()
#         #self.tracker = cv2.TrackerKCF_create()
#
#         #self.bbox = cv2.selectROI("Select Object", frame, False)
#         #cv2.destroyWindow("Select Object")
#         #self.tracker.init(frame, self.bbox)
#         # Create a canvas or label to display the video
    #     self.canvas = tk.Canvas(parent,bg="#262626", width=330, height=330)
    #     self.canvas.pack()
    #
    #     self.update()
    #
    # def update(self):
    #     ret, frame = self.vid.read()
    #     if not ret or frame is None:
    #         self.after_id = self.parent.after(self.delay, self.update)
    #         return
    #
    #     # Rotate frame 90 degrees counter-clockwise
    #     rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #
    #     # Use rotated frame for all processing
    #     grayFrame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
    #     blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    #
    #     # Detect circles
    #     circles = cv2.HoughCircles(
    #         blurFrame,
    #         cv2.HOUGH_GRADIENT,
    #         1.2,
    #         100,
    #         param1=100,
    #         param2=40,
    #         minRadius=75,
    #         maxRadius=300
    #     )
    #
    #     chosen = None
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         for i in circles[0, :]:
    #             if chosen is None:
    #                 chosen = i
    #             else:
    #                 # Track the circle closest to previous position
    #                 if self.dist(i[0], i[1], self.previousCircle[0], self.previousCircle[1]) < \
    #                 self.dist(chosen[0], chosen[1], self.previousCircle[0], self.previousCircle[1]):
    #                     chosen = i
    #
    #         # Draw the detected circle on rotated_frame
    #         cv2.circle(rotated_frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
    #         cv2.circle(rotated_frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
    #
    #     # Only update tracking and UI if a ball is detected
    #     if chosen is not None:
    #         x, y, r = chosen
    #         self.cx = x
    #         self.cy = y
    #
    #         # Update app targets
    #         self.app.target_x = int(self.cx)
    #         self.app.target_y = int(self.cy)
    #
    #         if self.app.panda_instance:
    #             self.app.panda_instance.update_target(self.cx, self.cy)
    #         if self.app.servo_instance:
    #             self.app.servo_instance.update_and_move_servo(self.app.target_x, self.app.target_y)
    #
    #         # Save for next frame tracking
    #         self.previousCircle = (self.cx, self.cy)
    #
    #         # Prepare frame for display
    #         canvas_width = self.canvas.winfo_width() or 330
    #         canvas_height = self.canvas.winfo_height() or 330
    #         display_frame = cv2.resize(rotated_frame, (canvas_width, canvas_height))
    #
    #         # Convert BGR to RGB for Tkinter
    #         cv2_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    #
    #         # Optional: Draw bounding box or marker
    #         x1_d = int((x - r) * canvas_width / rotated_frame.shape[1])
    #         y1_d = int((y - r) * canvas_height / rotated_frame.shape[0])
    #         cv2.circle(cv2_frame, (x1_d, y1_d), 5, (0, 0, 255), -1)
    #
    #         # Convert to Tkinter-compatible format
    #         img = Image.fromarray(cv2_frame)
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
    #         self.canvas.imgtk = imgtk
    #
    #         # Update coordinate label
    #         if self.coord_var:
    #             self.coord_var.set(f"Position: X: {self.cx}, Y: {self.cy}")
    #
    #     else:
    #         # No ball detected: optionally show frame without tracking, or clear display
    #         canvas_width = self.canvas.winfo_width() or 330
    #         canvas_height = self.canvas.winfo_height() or 330
    #         display_frame = cv2.resize(rotated_frame, (canvas_width, canvas_height))
    #         cv2_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    #
    #         # Optionally: indicate no detection (e.g., gray out or show message)
    #         cv2.putText(cv2_frame, "No ball detected", (10, 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    #         img = Image.fromarray(cv2_frame)
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
    #         self.canvas.imgtk = imgtk
    #
    #         # Clear coordinate display if desired
    #         if self.coord_var:
    #             self.coord_var.set("Position: --, --")
    #
    #     # Schedule next frame
    #     self.after_id = self.parent.after(self.delay, self.update)
