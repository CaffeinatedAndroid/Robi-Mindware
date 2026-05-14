import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from panda3d.core import AmbientLight, Vec4
from panda3d.core import Point3
from panda3d.core import PointLight, VBase4
from panda3d.core import TextureStage
from panda3d.core import DirectionalLight, NodePath
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from panda3d.core import Point2
from panda3d.core import Vec3
import cv2

import serial
import time



class MainApplication:
    def __init__(self, root):
        self.root = root
        #self.root.geometry("400x300")
        self.root.title("Panda3D in Tkinter PanedWindow")
        self.tk = self.root
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.tk['bg'] = '#262626'
        self.tk.title("Robo Mindware")

        #NOTE WEIGTH THE GRID TO SACLE WITH RATIO
        self.tk.grid_rowconfigure(1, weight=1)
        self.tk.grid_columnconfigure(0, weight=3)

        #NOTE CONTAINER AND FRAME FOR PANDA 3D RENDERER
        self.render_container = tk.LabelFrame(self.tk, text='World View', fg="white", bg="#262626")
        self.render_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.panda_frame = tk.LabelFrame(self.render_container, text='', fg="white", bg="#262626", width=380, height=340)
        self.panda_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        #NOTE START TOOLBAR CLASSES
        self.control_panel = ControlPanel(self)
        self.control_panel.grid(row=1, column=2, sticky="nsew")

        #NOTE SERVO COMMUNICATION CLASS
        self.servo_panel = tk.LabelFrame(self.tk, text='Servo', fg="white", bg="#262626")
        self.servo_panel.grid(row=2, column=2, sticky="nsew")

        toolbar_top_Frame = ToolBarTop()
        toolbar_top_Frame.grid(row=0, column=0, columnspan=3, sticky="new", padx=0, pady=0)

        toolbar_bottom_Frame = ToolBarBottom()
        toolbar_bottom_Frame.grid(row=5, column=0, columnspan=3,sticky="ew")

        #NOTE OPEN CV CONTAINER
        self.video_container = tk.LabelFrame(self.tk, text='Open CV', fg="white", bg="#262626")
        self.video_container.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.opencv_frame = tk.LabelFrame(self.video_container, text='', fg="white", bg="#262626", width=380, height=340)
        self.opencv_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # video_frame2 = StatFeed()
        # video_frame2.grid(row=1, column=2, sticky="nsew", padx=5, pady=2)

        # video_frame3 = MiscFeed()
        # video_frame3.grid(row=2, column=0, columnspan=3, rowspan=1, sticky="nsew", padx=5, pady=0)

        #NOTE PANDA 3D TOOLBAR
        button_Frame = ButtonBar(self.panda_frame, self)
        button_Frame.grid(row=1, column=0, sticky="ne", padx=15, pady=30)

        #NOTE OPENCV OBJECT POSITION
        self.coord_var = tk.StringVar()
        self.target_x = 0
        self.target_y = 0
        self.coord_var.set("Position: X: 0, Y: 0")
        coord_label = tk.Label(root, textvariable=self.coord_var, fg="white", bg="black")
        coord_label.grid(row=1, column=1, sticky="ne", padx=15, pady=30)

        #NOTE INSTANCES
        self.panda_instance = None
        self.opencv_instance = None
        self.servo_instance = None


    #NOTE CREATE PANDA 3D INSTANCE
    def load_panda3d(self):
        if self.panda_instance is None:
            # Ensure geometry is updated so winfo_id is valid
            self.tk.update_idletasks()
            #self.VideoFeed.update_frame()

            # Instantiate the nested Panda3dViewer class
            # Pass the Tkinter frame widget as the container
            self.panda_instance = Panda3dViewer(self.panda_frame)
            #self.panda_instance.look_at(self.x, self.y)

            # Start the Panda3D main loop running in the Tkinter event loop
            # Note: In some setups, you may need to call self.panda_instance.run()
            # or handle the loop via base.spawnTkLoop() depending on the specific
            # Panda3D version and Tkinter integration method.
            # For direct embedding, often the Tkinter mainloop suffices if
            # ShowBase was initialized with startTk().

    #NOTE DESTROY OPENCV INSTANCE
    def disable_panda3d(self):
        if self.panda_instance is not None:
            # Ensure geometry is updated so winfo_id is valid
            #self.VideoFeed.update_frame()
            self.panda_instance.destroy()
            self.panda_instance = None
            self.tk.update_idletasks()

    #NOTE CREATE OPENCV INSTANCE
    def StartServo(self):
        if self.servo_instance is None:
            self.servo_instance = ServoCommunication(self.servo_panel, self.target_x)
            self.tk.update_idletasks()


    #NOTE CREATE OPENCV INSTANCE
    def StartOpenCV(self):
        if self.opencv_instance is None:
            self.tk.update_idletasks()
            self.opencv_instance = VideoFeed(self.opencv_frame, self, video_source=1, delay=30, coord_var= self.coord_var)
            self.x = self.opencv_instance.cx
            self.y = self.opencv_instance.cy

    #NOTE DESTROY OPENCV INSTANCE
    def StopOpenCV(self):
        if self.opencv_instance is not None:

            self.opencv_instance.stop()
            if self.opencv_instance:
                self.opencv_instance.stop()  #NOTE Should cancel after() and release camera
                self.opencv_instance = None
            self.tk.update_idletasks()


    def change_view(self):
        if self.panda_instance:
            self.panda_instance.change_Camview()


class ServoCommunication(tk.Frame):
    def __init__(self, parent_tk_widget, target_x = None):
        super().__init__(parent_tk_widget)

        self.target = target_x
        # Initialize serial connection
        self.parent = parent_tk_widget

        super().__init__()
        self['bg'] = '#262626'

       #NOTE CELL TO PACK LOCALLY
        self.render_container = tk.Frame(self)
        self.render_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        #container_frame.config( height = 1, width = 20 )

        #NOTE DISABLE PANDA 3D FRAME
        self.descriptionLabel = tk.Label(self.render_container, text="Enter Command")
        self.descriptionLabel.pack(side=tk.TOP, padx=5, pady=5)


       #NOTE ACTIVATE 3D WINDOW, CALLS DEF FROM MAIN APPLICATION PARENT
        self.xAgleEntry = tk.Entry(self.render_container)
        self.xAgleEntry.pack(side=tk.TOP, padx=5, pady=5)


       #NOTE ENABLE OPENCV INSTANCE
        self.sendButton = tk.Button(self.render_container, text="Send", command= self.servo_comms)
        self.sendButton.config( height = 1, width = 20 )
        self.sendButton.pack(side=tk.TOP, padx=5, pady=5)

        self.arduino = serial.Serial('COM7', 9600, timeout=10)
        #time.sleep(2) # Wait for Arduino to reset
        #angle = self.xAgleEntry.get()

    def servo_comms(self, target_x):

        frame_width = 640

        normalized = (target_x - frame_width/2) / (frame_width/2)

        angle = int(90 + normalized * 90)

        angle = max(0, min(180, angle))

        self.arduino.write(f"{angle}\n".encode())

        print(f"Sent: {angle}")





#NOTE PANDA 3D TOOLBAR
class ButtonBar(tk.Frame):
    def __init__(self, parent_tk_widget, root):
        super().__init__()
        self.parent = root

       #NOTE CELL TO PACK LOCALLY
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=2, sticky="nsew")
        container_frame['bg'] = '#262626'

       #NOTE OPEN TICKETS TODO
        self.openLabel = tk.Button(container_frame, text="View X")
        self.openLabel.pack(side=tk.TOP)


       #NOTE CHANGE CAM VIEW

        self.closedLabel = tk.Button(container_frame, text="View Y",command = self.parent.change_view)
        self.closedLabel.pack(side=tk.TOP)


       #NOTE PENDING PICKUP TODO
        self.pendingLabel = tk.Button(container_frame, text="View Z")
        self.pendingLabel.pack(side=tk.TOP)


#NOTE CONTAINS DROP DOWN MENUS
class ToolBarTop(tk.Frame):
    def __init__(self):
        super().__init__()
        style = ttk.Style()

        # Configure the main widget appearance
        style.configure("Custom.TCombobox",
            fieldbackground="#3a3a3a",  # Entry background
            background="#3a3a3a",       # Button background
            foreground="#ffffff",       # Text color
            arrowcolor="#007fff"        # Arrow color
        )

        # Map state-based changes (e.g., hover, focus)
        style.map("Custom.TCombobox",
            fieldbackground=[("readonly", "#000000"), ("focus", "#000000")],
            foreground=[("readonly", "#ffffff"), ("focus", "#000000")]
        )

        self.option_add("*TCombobox*Listbox.background", "#3a3a3a")
        self.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        self.option_add("*TCombobox*Listbox.selectBackground", "#007fff")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

        layout = [
            ('Combobox.button', {
                'sticky': 'w',
                'children': [
                            ('Combobox.textarea', {
                                'sticky': 'w'
                            })
                        ]
                    })
        ]

        style.layout("Custom.TCombobox", layout)

        self['bg'] = '#3a3a3a'

        options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]

        style.configure("Custom.TCombobox", postoffset=(0, 0, 50, 0))

        def on_select( event):
        # Reset the Combobox to the default title/value after selection
            self.fileMenu.set("File")
            self.settingMenu.set("Settings")
            self.windowsMenu.set("Window")

       #NOTE CELL TO PACK LOCALLY
        container_frame = tk.Frame(self)
        container_frame.grid(row=0, column=0, sticky="ew")
        container_frame['bg'] = '#3a3a3a'

       #NOTE FILE MENU DROP DOWN TODO
        self.fileMenu = ttk.Combobox(container_frame, state="readonly", text="File", values=options, style="Custom.TCombobox", width=4)
        self.fileMenu.pack(side=tk.LEFT, padx=0)
        self.fileMenu.configure(state="readonly")
        self.fileMenu.set("File")
        self.fileMenu.bind("<<ComboboxSelected>>", on_select)

       #NOTE SETTINGS MENU DROP DOWN TODO
        self.settingMenu = ttk.Combobox(container_frame, state="readonly", text="Settings", values=options, style="Custom.TCombobox", width=8)
        self.settingMenu.pack(side=tk.LEFT, padx=0)
        self.settingMenu.configure(state="readonly")
        self.settingMenu.set("Settings")
        self.settingMenu.bind("<<ComboboxSelected>>", on_select)

       #NOTE WINDOW MENU DROPDOWN TODO
        self.windowsMenu = ttk.Combobox(container_frame, state="readonly", text="Window", values=options, style="Custom.TCombobox", width=9)
        self.windowsMenu.pack(side=tk.LEFT, padx=0)
        self.windowsMenu.configure(state="readonly")
        self.windowsMenu.set("Window")
        self.windowsMenu.bind("<<ComboboxSelected>>", on_select)



class ToolBarBottom(tk.Frame): #TODO
    def __init__(self):
        super().__init__()
        self['bg'] = '#3a3a3a'

       #NOTE CELL TO PACK LOCALLY
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=0, sticky="ew")
        container_frame['bg'] = '#3a3a3a'

       #NOTE OPEN TICKETS TODO
        self.openLabel = tk.Label(container_frame, fg="white", bg="#3a3a3a", text="Open Tickets:")
        self.openLabel.pack(side=tk.LEFT, padx=0)


       #NOTE CLOSED TICKETS TODO
        self.closedLabel = tk.Label(container_frame, fg="white", bg="#3a3a3a", text="Closed Tickets:")
        self.closedLabel.pack(side=tk.LEFT, padx=0)


       #NOTE PENDING PICKUP TODO
        self.connectionLabel = tk.Label(container_frame, fg="red", bg="#3a3a3a", text="Disconnected")
        self.connectionLabel.pack(side=tk.RIGHT, padx=5)


class ControlPanel(tk.Frame):
    def __init__(self, parent_tk_widget):

        super().__init__()
        self['bg'] = '#262626'

       #NOTE CELL TO PACK LOCALLY
        self.render_container = tk.LabelFrame(self, text='Control Panel', fg="white", bg="#262626")
        self.render_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        #container_frame.config( height = 1, width = 20 )

       #NOTE ACTIVATE 3D WINDOW, CALLS DEF FROM MAIN APPLICATION PARENT
        self.openLabel = tk.Button(self.render_container, text="Enable 3D View", command= parent_tk_widget.load_panda3d)
        self.openLabel.config( height = 1, width = 20 )
        self.openLabel.pack(side=tk.TOP, padx=5, pady=5)

        #NOTE DISABLE PANDA 3D FRAME
        self.openLabel = tk.Button(self.render_container, text="Disable 3D View", command= parent_tk_widget.disable_panda3d)
        self.openLabel.config( height = 1, width = 20 )
        self.openLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE ENABLE OPENCV INSTANCE
        self.closedLabel = tk.Button(self.render_container, text="Enable OpenCV", command= parent_tk_widget.StartOpenCV)
        self.closedLabel.config( height = 1, width = 20 )
        self.closedLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE CLOSE OPENCV INSTANCE
        self.pendingLabel = tk.Button(self.render_container, text="Disable OpenCV", command= parent_tk_widget.StopOpenCV)
        self.pendingLabel.config( height = 1, width = 20 )
        self.pendingLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE CLOSE OPENCV INSTANCE
        self.pendingLabel = tk.Button(self.render_container, text="Connect Servo", command= parent_tk_widget.StartServo)
        self.pendingLabel.config( height = 1, width = 20 )
        self.pendingLabel.pack(side=tk.TOP, padx=5, pady=5)





class VideoFeed(): #NOTE OPEN CV MANAGER

    def __init__(self, parent,app, video_source=1, delay=30, coord_var= None):
        self.app = app
        self.parent = parent
        self.delay = delay
        self.vid = cv2.VideoCapture(video_source)
        self.coord_var = coord_var
        ret, frame = self.vid.read()
        self.tracker = cv2.TrackerKCF_create()

        self.bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        self.tracker.init(frame, self.bbox)
        # Create a canvas or label to display the video
        self.canvas = tk.Canvas(parent,bg="#262626", width=330, height=330)
        self.canvas.pack()

        self.update()

    def update(self):
        ret, frame = self.vid.read()
        if not ret or frame is None:
            return

        # Use original frame for tracking
        ret, self.bbox = self.tracker.update(frame)
        if ret:
            x, y, w, h = [int(v) for v in self.bbox]
            self.cx = x + w // 2
            self.cy = frame.shape[0] - (y + h // 2)

            self.app.target_x = self.cx
            self.app.target_y = self.cy

            if self.app.panda_instance:
                self.app.panda_instance.update_target(self.cx, self.cy)
            if self.app.servo_instance:
                self.app.servo_instance.servo_comms(self.app.target_x)

            # Resize only for display
            canvas_width = self.canvas.winfo_width() or 330
            canvas_height = self.canvas.winfo_height() or 330
            display_frame = cv2.resize(frame, (canvas_width, canvas_height))

            # Scale the bounding box for display
            x_disp = int(x * canvas_width / frame.shape[1])
            y_disp = int(y * canvas_height / frame.shape[0])
            w_disp = int(w * canvas_width / frame.shape[1])
            h_disp = int(h * canvas_height / frame.shape[0])

            cv2_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(cv2_frame, (x_disp, y_disp), (x_disp + w_disp, y_disp + h_disp), (0, 255, 0), 2)
            cv2.circle(cv2_frame, (x_disp, y_disp), 5, (0, 0, 255), -1)

            img = Image.fromarray(cv2_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
            self.canvas.imgtk = imgtk

            if self.coord_var:
                self.coord_var.set(f"Position: X: {self.cx}, Y: {self.cy}")


        self.after_id = self.parent.after(self.delay, self.update)
        #self.after_id = self.parent.after(self.delay, self.update)
        # Schedule the next update
        #self.parent.after(self.delay, self.update)

    def stop(self):

        if self.after_id:
            self.parent.after_cancel(self.after_id)

        self.vid.release()
        cv2.destroyAllWindows()
        self.canvas.destroy()






class Panda3dViewer(ShowBase):
    """
    Nested class that handles Panda3D initialization and rendering.
    Inherits from ShowBase to access rendering capabilities.
    """
    def __init__(self, parent_tk_widget):
        # 1. Initialize ShowBase without opening a window
        ShowBase.__init__(self, windowType='none')
        self.startTk()

        props = WindowProperties()
        props.set_parent_window(parent_tk_widget.winfo_id())
        props.set_origin(0, 0)
        props.set_size(parent_tk_widget.winfo_width(), parent_tk_widget.winfo_height())

        self.make_default_pipe()
        self.open_default_window(props=props)

        #parent_tk_widget.bind("<Configure>", self.on_resize) #TODO FIX

        self.setBackgroundColor(0, 0, 0.1)

        #NOTE 3D ROBOT SETUP

        self.head = self.loader.loadModel("RobiHead.glb")
        self.head.setHpr(self.render, 90, 90, 90)
        # Add the model to the scene graph and position it
        self.head.reparentTo(self.render)
        self.head.setPos(Point3(0, 0, 0))
        self.head.setScale(50)

        self.neck = self.loader.loadModel("RobiNeck.glb")
        self.neck.setHpr(self.render, 90, 90, 90)
        # Add the model to the scene graph and position it
        self.neck.reparentTo(self.render)
        self.neck.setPos(Point3(0, 0, 0))
        self.neck.setScale(50)



        self.body = self.loader.loadModel("RobiStaticBody.glb")
        self.body.setHpr(self.render, 90, 90, 90)
        # Add the model to the scene graph and position it
        self.body.reparentTo(self.render)
        self.body.setPos(Point3(0, 0, 0))
        self.body.setScale(50)

        self.plane = self.loader.loadModel("plane.glb")
        tex = loader.loadTexture('GridGrey1.png')
        ts = TextureStage("ts")
        self.plane.setTexture(ts, tex)
        self.plane.setTexScale(ts, 4, 4)
        self.plane.setHpr(self.render, 0, 0, 0)
        self.plane.reparentTo(self.render)
        self.plane.setPos(Point3(0, 0, 0))
        self.plane.setScale(5)

        self.base_pivot = self.render.attachNewNode("base_pivot")
        self.base_pivot.setPos(0, 0, 0)
        self.neck_pivot = self.base_pivot.attachNewNode("Neck_pivot")
        self.neck_pivot.setPos(0, 0, 0)
        self.head_pivot = self.neck_pivot.attachNewNode("head_pivot")
        self.head_pivot.setPos(0, 0, 0)

        self.neck.reparentTo(self.neck_pivot)
        self.body.reparentTo(self.base_pivot)
        self.head.reparentTo(self.head_pivot)

        # BODY
        # self.body_offset = self.base_pivot.attachNewNode("body_offset")
        # self.body.reparentTo(self.body_offset)
        #
        # # NECK
        # self.neck_offset = self.neck_pivot.attachNewNode("neck_offset")
        # self.neck.reparentTo(self.neck_offset)
        #
        # # HEAD
        # self.head_offset = self.head_pivot.attachNewNode("head_offset")
        # self.head.reparentTo(self.head_offset)

        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor(Vec4(0.25, 0.25, 0.25, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        plight = DirectionalLight("spotlight")
        plight.setColor(VBase4(1.0, 1.0, 1.0, 2.0))  # White light
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(self.base_pivot, 0, 0, -25)  # Set position (x, y, z)
        plnp.node().setScene(self.render)
        plnp.setHpr(-200, -50, -270)
        self.render.setLight(plnp)

        self.render.setShaderAuto()

        # Move the camera to a new position
        #self.disableMouse()
        self.camera.setPos(100, 100, 2)
        base.camLens.setNearFar(1.0, 500.0)
        self.camera.lookAt(self.body, -0.2, 0.2, 0.2)  # Make the camera look at the cube
        #self.enableMouse()
        #self.taskMgr.add(self.pivot, "spinCubeTask")


    # def spin_cube(self, task):
    #      # Calculate angle based on time (60 degrees per second)
    #     angle = task.time * 25
    #      # Rotate around all axes
    #     self.pivot.setHpr(angle, 0, 0)
    #     return task.cont


    #NOTE TODO RE IMPLEMENT
    # def on_resize(self, event):
    #     # Update frame geometry to ensure accurate dimensions
    #     self.tk.update_idletasks()
    #
    #     props = WindowProperties()
    #     props.setOrigin(0, 0)
    #     props.setSize(event.width, event.height)
    #
    #     if self.win:
    #         self.win.requestProperties(props)

    def change_Camview(self): #NOTE TESTING
        #self.taskMgr.remove('spinCubeTask')
        self.camera.setPos(40, -35, -40)
        base.camLens.setNearFar(1.0, 500.0)
        self.camera.lookAt(self.cube, 0, 0.2, 0)
        return

    def update_target(self, cx, cy):

        frame_width = 640
        frame_height = 480

        nx = (cx - frame_width/2) / (frame_width/2)
        ny = (cy - frame_height/2) / (frame_height/2)

        heading = nx * 70
        pitch = ny * 30

        # Smooth motion
        current_h = self.neck_pivot.getH()
        current_p = self.head_pivot.getP()

        smooth_h = current_h + (heading - current_h) * 0.15
        smooth_p = current_p + (-pitch - current_p) * 0.15

        self.neck_pivot.setH(smooth_h)
        self.head_pivot.setP(smooth_p)



if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
