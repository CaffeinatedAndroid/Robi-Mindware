from tkinter import *
import tkinter as tk
from tkinter import Button
from tkinter import ttk
import tkinter.ttk as ttk
from direct.showbase.ShowBase import ShowBase

from panda3d.core import AmbientLight, Vec4
from panda3d.core import Point3
from panda3d.core import WindowProperties
from panda3d.core import PointLight, VBase4
from panda3d.core import TextureStage
from panda3d.core import DirectionalLight, NodePath

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
import cv2


from PIL import Image, ImageTk


#NOTE TO KEEP LOGICAL ORDER OF CODE, TOP TO BOTTOM
def main():
    app = Application()
    app.run()

#_______________________________________________________________ ROOT CLASS _______________________________________________________________#
#NOTE INITIALISES APPLICATION ROOT AND SETS UP FRAMES
class Application(ShowBase):
    def __init__(self):
        ShowBase.__init__(self, windowType='none') #NOTE START PANDA 3D WITH NO DEFAULT WINDOW
        self.startTk()

       #NOTE MAKE FULL SCREEN BY DEFAULT, NOT THE SAME AS .attributes(-Fullscreen, bool)
        #self.state("zoomed")
        self.tk = self.tkRoot
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.tk['bg'] = '#262626'
        self.tk.title("Robo Mindware")
       #NOTE WEIGTH THE GRID TO SACLE WITH RATIO
        #self.tk.grid_rowconfigure(1, weight=1)
        #self.tk.grid_columnconfigure(0, weight=3)
        self.tk.grid_columnconfigure(2, weight=3)
        self.tk.grid_columnconfigure(3, weight=1)

       #NOTE START TOOLBAR CLASSES
        control_panel = ControlPanel()
        control_panel.grid(row=1, column=3, sticky="nsew", padx=0, pady=13)


        filler_Frame = FillerBar()
        filler_Frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        filler_Frame2 = FillerBar()
        filler_Frame2.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)



        toolbar_top_Frame = ToolBarTop()
        toolbar_top_Frame.grid(row=0, column=0, sticky="new", padx=3, pady=3)

        filler_Frame3 = FillerBar()
        filler_Frame3.grid(row=5, column=0, sticky="nsew", padx=0, pady=0)

        filler_Frame4 = FillerBar()
        filler_Frame4.grid(row=5, column=1, sticky="nsew", padx=0, pady=0)


        filler_Frame5 = FillerBar()
        filler_Frame5.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)

        filler_Frame6 = FillerBar()
        filler_Frame6.grid(row=5, column=2, sticky="nsew", padx=0, pady=0)



        toolbar_bottom_Frame = ToolBarBottom()
        toolbar_bottom_Frame.grid(row=5, column=0, sticky="ew")


        video_frame = VideoFeed()
        video_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=2)

        video_frame2 = StatFeed()
        video_frame2.grid(row=1, column=2, sticky="nsew", padx=5, pady=2)

        video_frame3 = MiscFeed()
        video_frame3.grid(row=2, column=0, columnspan=3, rowspan=1, sticky="nsew", padx=5, pady=0)

        #NOTE CREATE AND PACK LABEL FRAME
        self.panda_frame = tk.LabelFrame(self.tk, text='World View', fg="white", bg="#262626", width=380, height=340)
        self.panda_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)


        button_Frame = ButtonBar()
        button_Frame.grid(row=1, column=0, sticky="ne", padx=10, pady=20)


        #NOTE Set properties of the Panda3D window
        props = WindowProperties()
        props.set_parent_window(self.panda_frame.winfo_id())  #NOTE Display within the label frame
        props.set_origin(5, 15)  #NOTE Relative to the label frame
        props.set_size(373, 340)

        self.make_default_pipe()
        self.open_default_window(props=props)


        self.setBackgroundColor(0, 0, 0.1)
        #NOTE SETUP SCENE DATA
        # scene = self.loader.load_model("environment")
        #
        # scene.reparent_to(self.render)
        # scene.setPos(Point3(0, 200, -50))

        self.cube = self.loader.loadModel("Robi.glb")
        self.cube.setHpr(self.render, 90, 90, 90)
        # Add the model to the scene graph and position it
        self.cube.reparentTo(self.render)
        self.cube.setPos(Point3(0, 0, -50))

        self.cube.setScale(50)


        self.plane = self.loader.loadModel("plane.glb")
        tex = loader.loadTexture('GridGrey1.png')
        ts = TextureStage("ts")
        self.plane.setTexture(ts, tex)
        self.plane.setTexScale(ts, 4, 4)
        self.plane.setHpr(self.render, 0, 0, 0)
        # Add the model to the scene graph and position it
        self.plane.reparentTo(self.render)
        self.plane.setPos(Point3(0, 0, -50))

        self.plane.setScale(5)





        self.pivot = self.render.attachNewNode("pivot")
        self.pivot.setPos(0, 0, 0)

        # Parent the model to the pivot
        self.cube.reparentTo(self.pivot)
        self.plane.reparentTo(self.pivot)



        # Create the light object and set its color (RGBA)
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor(Vec4(0.25, 0.25, 0.25, 1))

        # Attach to scene graph
        ambientLightNP = self.render.attachNewNode(ambientLight)

        # Enable for the entire scene
        self.render.setLight(ambientLightNP)

        plight = DirectionalLight("spotlight")
        plight.setColor(VBase4(1.0, 1.0, 1.0, 2.0))  # White light


        # 2. Create a NodePath for the light and attach it to the scene
        plnp = render.attachNewNode(plight)
        plnp.setPos(self.pivot, 0, 0, -25)  # Set position (x, y, z)
        plnp.node().setScene(self.render)
        plnp.setHpr(-200, -50, -270)
        # 3. Apply the light to the scene graph
        # Option A: Illuminate everything
        #plnp.node().setShadowCaster(True, 128, 128)
        self.render.setLight(plnp)


        # Enable for the entire scene
        self.render.setShaderAuto()

        # Move the camera to a new position
        self.disableMouse()
        self.camera.setPos(40, 40, -40)
        base.camLens.setNearFar(1.0, 500.0)
        self.camera.lookAt(self.cube, -0.2, 0.2, 0.2)  # Make the camera look at the cube
        #self.enableMouse()
        self.taskMgr.add(self.spin_cube, "spinCubeTask")

    def spin_cube(self, task):
         # Calculate angle based on time (60 degrees per second)
        angle = task.time * 25
         # Rotate around all axes
        self.pivot.setHpr(angle, 0, 0)
        return task.cont



#_______________________________________________________________ TOOLBAR BOTTOM CLASS TODO _______________________________________________________________#

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






class ToolBarBottom(tk.Frame):
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
        self.pendingLabel = tk.Label(container_frame, fg="white", bg="#3a3a3a", text="Pending Pickup:")
        self.pendingLabel.pack(side=tk.LEFT, padx=0)





#_______________________________________________________________ BUTTON BAR CLASS TODO _______________________________________________________________#

class ButtonBar(tk.Frame):
    def __init__(self):
        super().__init__()

       #NOTE CELL TO PACK LOCALLY
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=2, sticky="nsew")
        container_frame['bg'] = '#262626'

       #NOTE OPEN TICKETS TODO
        self.openLabel = tk.Button(container_frame, text="View X")
        self.openLabel.pack(side=tk.TOP)


       #NOTE CLOSED TICKETS TODO
        self.closedLabel = tk.Button(container_frame, text="View Y")
        self.closedLabel.pack(side=tk.TOP)


       #NOTE PENDING PICKUP TODO
        self.pendingLabel = tk.Button(container_frame, text="View Z")
        self.pendingLabel.pack(side=tk.TOP)




class ControlPanel(tk.Frame):
    def __init__(self):

        super().__init__()
        self['bg'] = '#262626'

       #NOTE CELL TO PACK LOCALLY
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        #container_frame.config( height = 1, width = 20 )
        container_frame['bg'] = '#262626'

       #NOTE OPEN TICKETS TODO
        self.openLabel = tk.Button(container_frame, text="Activate")
        self.openLabel.config( height = 1, width = 20 )
        self.openLabel.pack(side=tk.TOP)


       #NOTE CLOSED TICKETS TODO
        self.closedLabel = tk.Button(container_frame, text="Servo Position")
        self.closedLabel.config( height = 1, width = 20 )
        self.closedLabel.pack(side=tk.TOP)


       #NOTE PENDING PICKUP TODO
        self.pendingLabel = tk.Button(container_frame, text="Config")
        self.pendingLabel.config( height = 1, width = 20 )
        self.pendingLabel.pack(side=tk.TOP)


#_______________________________________________________________ VIDEO PANEL CLASS TODO _______________________________________________________________#

class VideoFeed(tk.Frame):
    def __init__(self):
        super().__init__()
        self['bg'] = '#262626'
        root = self
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        final = ImageTk.PhotoImage(img)
        #imgtk = ImageTk.PhotoImage(image=img)
        # image = Image.open("image.png")
        video_label = tk.Label(root, width=407, height=317, bg="#000000", image=final)
        video_label.grid(row=1, column=0, sticky="nsew")
        # # 2. Convert it to a Tkinter-compatible format
        video_label.image = final

        # label = tk.Label(root, width=407, height=317, bg="#000000", image=photo)
        # label.place(x=5, y=15)
        #
        # # IMPORTANT: Keep a reference to prevent garbage collection
        # label.image = photo

        #video_label.imgtk = imgtk  # Save reference to prevent garbage collection
       # video_label.configure(image=imgtk)
            #root.after(3000)  # Call every 20ms



        video_label.pack()

        #self.show_frame()




class MiscFeed(tk.Frame):
    def __init__(self):
        super().__init__()
        self['bg'] = '#262626'
        root = self
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=0, sticky="nsew")
        container_frame['bg'] = '#3a3a3a'

        video_label = tk.LabelFrame(container_frame, text='Misc Feed', fg="white", bg="#262626", width=400, height=300)
        video_label.grid(row=1, column=0, sticky="nsew")
        # # 2. Convert it to a Tkinter-compatible format
        # photo = ImageTk.PhotoImage(image)
        #
        # # 3. Create a label to display the image
        # label = tk.Label(root, width=407, height=317, bg="#000000", image=photo)
        # label.place(x=5, y=15)
        #
        # # IMPORTANT: Keep a reference to prevent garbage collection
        # label.image = photo



        fig = Figure(figsize=(22, 4), dpi=53)
        ax = fig.add_subplot(1, 1, 1)

        # 3. Plot Data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, color='red', label='sin(x)')
        ax.plot(x , y*2, color='blue', label='sin(x)')
        ax.set_title("Charge")
        ax.legend()
        ax.set_facecolor('#000000')
        ax.grid(True, color='grey', linewidth=1.4, linestyle='-.')

        fig.tight_layout()
        fig.patch.set_facecolor('#3a3a3a')
        ax.tick_params(axis='x', colors='white')       # X-axis tick labels
        ax.tick_params(axis='y', colors='white')      # Y-axis tick labels

        #
        ax.set_xlabel('X Axis', color='white')
        ax.set_ylabel('Y Axis', color='white')


        # 3. Embed the Figure into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=video_label)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady =0)


        # toolbar = NavigationToolbar2Tk(canvas, container_frame)
        # toolbar.update()
        # canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        # toolbar.autoscale(False)


class StatFeed(tk.Frame):
    def __init__(self):
        super().__init__()
        self['bg'] = '#262626'
        root = self
        container_frame = tk.Frame(self)
        container_frame.grid(row=1, column=0, sticky="nsew")
        container_frame['bg'] = '#3a3a3a'

        video_label = tk.LabelFrame(container_frame, text='Video Feed', fg="white", bg="#262626", width=440, height=360)
        video_label.grid(row=1, column=0, sticky="nsew")
        # # 2. Convert it to a Tkinter-compatible format
        # photo = ImageTk.PhotoImage(image)
        #
        # # 3. Create a label to display the image
        # label = tk.Label(root, width=407, height=317, bg="#000000", image=photo)
        # label.place(x=5, y=15)
        #
        # # IMPORTANT: Keep a reference to prevent garbage collection
        # label.image = photo



        fig = Figure(figsize=(6, 5.6), dpi=60)
        #
        #
        ax = fig.add_subplot(111, projection='3d')
        fig.set_facecolor('#020314')
        ax.margins(x=0, y=0)
        ax.set_facecolor('#020314')
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.tick_params(axis='x', colors='white')       # X-axis tick labels
        ax.tick_params(axis='y', colors='white')      # Y-axis tick labels
        ax.tick_params(axis='z', colors='white')      # Y-axis tick labels
        #
        ax.set_xlabel('X Axis', color='white')
        ax.set_ylabel('Y Axis', color='white')
        ax.set_zlabel('Z Axis', color='white')
        ax.set_xlim(0, 2)   # Set x-axis range
        ax.set_ylim(0, 2) # Set y-axis range
        ax.set_zlim(0, 2) # Set y-axis range
        #
        x = np.random.rand(100) * 2
        y = np.random.rand(100) * 2
        z = np.random.rand(100) * 2
        ax.scatter(x, y, z, c='r', marker='o')
        #
        # 3. Embed the Figure into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=video_label)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady =2)


        # toolbar = NavigationToolbar2Tk(canvas, container_frame)
        # toolbar.update()
        # canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        # toolbar.autoscale(False)


#_______________________________________________________________ GRAPH CLASS TODO _______________________________________________________________#

class FillerBar(tk.Frame):
    def __init__(self):
        super().__init__()


        self.filler = tk.Label(self, text='', bd='2', fg="#3a3a3a", bg="#3a3a3a")
        self.filler.pack(fill = BOTH, expand = True)



#_______________________________________________________________   START APP   _______________________________________________________________#
if __name__ == "__main__":
    main()

