import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


 #TODO MAKE SURE THE APPLICATION SHUTS DOWN WITH SHUTDOWN BUTTON
 #TODO MAKE SURE MAIN RUNS TERMINAL IN BACKGROUND TO ENABLE PROPER FEEDBACK
class UIManager():

    def __init__(self, bus):

        self.queue = bus.subscribe(
            "target_detected"
        )

        self.frame_queue = bus.subscribe(
            "vision_frame",
            maxsize=1
        )

        self.root = tk.Tk()
        self.tk = self.root
        # self.style = ttk.Style()
        # self.style.theme_use('clam')
        self.tk['bg'] = '#262626'

        self.tk.title("MINDWARE MANAGER")

        self.poll_bus()
        self.poll_frames()

        #self.tk.geometry("400x300")
        self.videoFrame = None



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

        self.opencv_frame = tk.Label(self.video_container)
        self.opencv_frame.pack(side=tk.LEFT, padx=0)

        self.busStatusLabel = tk.Label(self.tk, text="Waiting...")

        self.busStatusLabel.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)


        #NOTE PANDA 3D TOOLBAR
        button_Frame = ButtonBar(self.panda_frame, self)
        button_Frame.grid(row=1, column=0, sticky="ne", padx=15, pady=30)

        #NOTE OBJECT POSITION
        self.coord_var = tk.StringVar()
        self.target_x = 0
        self.target_y = 0
        self.coord_var.set("Position: X: 0, Y: 0")
        self.coord_label = tk.Label(self.tk, textvariable=self.coord_var, fg="white", bg="black")
        self.coord_label.grid(row=1, column=1, sticky="ne", padx=15, pady=30)






    def run(self):
        self.root.mainloop()

    def quit(self):
        self.root.quit()



    def poll_frames(self):

        try:

            while True:

                envelope = self.frame_queue.get_nowait()

                if envelope is None:
                    return

                frame_msg = envelope.payload

                frame = frame_msg.frame

                # Convert NumPy array -> PIL image
                image = Image.fromarray(frame)

                # Convert PIL -> Tkinter image
                photo = ImageTk.PhotoImage(image)

                # Update label
                self.opencv_frame.config(image=photo)

                # Prevent garbage collection
                self.opencv_frame.image = photo

        except queue.Empty:
            pass

        self.root.after(30, self.poll_frames)

    def poll_bus(self):

        try:

            while True:

                envelope = self.queue.get_nowait()

                if envelope is None:
                    self.root.quit()
                    return

                msg = envelope.payload

                self.coord_var.set( f"(X:{msg.x}, Y:{msg.y})")
                #self.videoFrame = msg.frame
                self.busStatusLabel.config(text = f"Data Recieved: Coordinates: {msg.x}, {msg.y}")


        except queue.Empty:
            #self.busStatusLabel.config(text = "No messages on Bus")
            pass

        # schedule next poll
        self.root.after(50, self.poll_bus)



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

        self.closedLabel = tk.Button(container_frame, text="View Y")
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
        self.openLabel = tk.Button(self.render_container, text="Enable 3D View")
        self.openLabel.config( height = 1, width = 20 )
        self.openLabel.pack(side=tk.TOP, padx=5, pady=5)

        #NOTE DISABLE PANDA 3D FRAME
        self.openLabel = tk.Button(self.render_container, text="Disable 3D View")
        self.openLabel.config( height = 1, width = 20 )
        self.openLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE ENABLE OPENCV INSTANCE
        self.closedLabel = tk.Button(self.render_container, text="Enable OpenCV")
        self.closedLabel.config( height = 1, width = 20 )
        self.closedLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE CLOSE OPENCV INSTANCE
        self.pendingLabel = tk.Button(self.render_container, text="Disable OpenCV")
        self.pendingLabel.config( height = 1, width = 20 )
        self.pendingLabel.pack(side=tk.TOP, padx=5, pady=5)

       #NOTE CLOSE OPENCV INSTANCE
        self.shutdown_Button = tk.Button(self.render_container, text="Shutdown")
        self.shutdown_Button.config( height = 1, width = 20 )
        self.shutdown_Button.pack(side=tk.TOP, padx=5, pady=5)



