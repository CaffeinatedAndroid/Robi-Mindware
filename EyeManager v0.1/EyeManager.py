from CTkColorPicker import CTkColorPicker
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import tkinter as tk
import numpy as np
import colorsys
import math
import json
import cv2
import os
import copy
import sys
# FOR DEBUGGING
import time


#BUILD COMMAND:
# pyinstaller \
#   --onefile \
#   --windowed \
#   --collect-data CTkColorPicker \
#   --collect-submodules PIL \
#   --hidden-import=PIL._tkinter_finder \
#   --add-data "resources:resources" \
#   EyeManager.py


def resource_path(*paths):
    """Return absolute path to resource, works for PyInstaller and source."""
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, *paths)

def res(filename):
    return resource_path("resources", filename)
print("Base path:", resource_path())
print("Left image:", res("lcdFrontLeft.png"))
print("Exists:", os.path.exists(res("lcdFrontLeft.png")))
start=time.time()


# CONFIGURE CUSTOM-TKINTER
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# GC9A01 SIMULATOR CONSTANTS
WIDTH = 240
HEIGHT = 240
CENTER = (WIDTH // 2, HEIGHT // 2)

PREVIEW_W = 125
PREVIEW_H = 160

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()

        #ADD RESOURCE HELPER AND MODIFY RESOURCE PATHS FOR BUILDING AS EXECUTABLE




        # NOTE MAIN APPLICATION CONFIGURATION
        self.title("Eye Manager v0.1")
        self.geometry("1750x900")

        # NOTE ANIMATION STATES
        self.y_pos = 0
        self.hue_cycle = 0
        self.angle = 0
        self.running = False
        self.going_down = True
        self.blinking = False

        self.anim_rotation = 0
        self.anim_eyelid_y = 0
        self.image = np.zeros((240, 240, 3), dtype=np.uint8)
        src = np.float32([
            [0, 0],
            [WIDTH, 0],
            [0, HEIGHT],
            [WIDTH, HEIGHT]
        ])


      # NOTE PREVIEW PANEL EYE WARP

        sx = PREVIEW_W / WIDTH
        sy = PREVIEW_H / HEIGHT

        dst_left = np.float32([
            [-20 * sx,                 0],
            [PREVIEW_W,               20 * sy],
            [0,                       PREVIEW_H - 20 * sy],
            [PREVIEW_W - 20 * sx,     PREVIEW_H]
        ])

        dst_right = np.float32([
            [0,                       20 * sy],
            [PREVIEW_W + 20 * sx,     0],
            [20 * sx,                 PREVIEW_H],
            [PREVIEW_W,               PREVIEW_H - 20 * sy]
        ])

        self.M_left = cv2.getPerspectiveTransform(src, dst_left)
        self.M_right = cv2.getPerspectiveTransform(src, dst_right)


# ______________________________________________________________________  NOTE SCENE OBJECTS ______________________________________________________________________ #
        self.scene = [
                {
                    "name": "eyelid",
                    "type": "circle",
                    "enabled": True,
                    "center": [120, 0],
                    "radius": 57,
                    "fill": True,
                    "color": [0,0,0],
                    "layer": 90,
                },

                {
                    "name": "iris",
                    "type": "ellipse",
                    "enabled": True,
                    "center": [120,120],
                    "axes": [40,55],
                    "fill": True,
                    "rainbow": True,
                    "color": [255,255,255],
                    "layer": 10,
                },

                {
                    "name": "spiral",
                    "type": "spiral",
                    "enabled": True,
                    "center": [120,120],
                    "radius": 45,
                    "turns": 5,
                    "rotation": 0,
                    "color": [255,255,255],
                    "thickness": 3,
                    "layer": 20,
                },

                {
                    "name": "pupil",
                    "type": "circle",
                    "enabled": True,
                    "center": [120,120],
                    "radius": 22,
                    "fill": True,
                    "rainbow": True,
                    "color": [255,255,255],
                    "layer": 30,
                },

                {
                    "name": "ring",
                    "type": "ring",
                    "enabled": True,
                    "center": [120,120],
                    "radius": 100,
                    "thickness": 20,
                    "rainbow": True,
                    "color": [255,255,255],
                    "layer": 100,
                }
            ]

        # NOTE LAYER SORTING INIT
        self.scene.sort(key=lambda o: o["layer"])

        self.objects = {
            obj["name"]: obj
            for obj in self.scene
        }

        # NOTE  COPIES SCENE AS DEFAULT FOR "clearScreen" FUNCTION
        self.default_scene = copy.deepcopy(self.scene)


        # NOTE PRECOMPUTE SPIRAL
        self.spiral_points = []

        spiral = self.getObject("spiral")

        for theta in np.linspace(0, spiral["turns"] * 2 * math.pi, 250):
            r = spiral["radius"] * (theta / (spiral["turns"] * 2 * math.pi))

            self.spiral_points.append([
                r * math.cos(theta),
                r * math.sin(theta)
            ])

        self.spiral_points = np.array(self.spiral_points, dtype=np.float32)


        # NOTE FOR PERFORMANCE DEBUGGING
        print("scene:", time.time()-start)

# ______________________________________________________________________ NOTE WINDOW LAYOUT ______________________________________________________________________ #

        # NOTE GENERAL GRID WEIGHTS
        self.grid_rowconfigure(1, weight=1)

        self.grid_columnconfigure(0, weight=1)                # Tool panel
        self.grid_columnconfigure(1, minsize=100, weight=0)   # Inspector
        self.grid_columnconfigure(2, weight=4)                # Editor
        self.grid_columnconfigure(3, weight=4)                # Preview


        # NOTE MAIN FRAMES / PANELS
        self.frame1 = ctk.CTkFrame(self,fg_color="gray25")
        self.frame1.grid(row=1, column=2, sticky="nsew", padx=1, pady=1)

        self.frame2 = ctk.CTkFrame(self,fg_color="gray25")
        self.frame2.grid(row=1, column=3, sticky="nsew", padx=1, pady=1)

        self.frame3 = ctk.CTkFrame(self,fg_color="gray25")
        self.frame3.grid(row=2, column=3, sticky="nsew", rowspan=2,columnspan=3,padx=1, pady=1)
        self.frame3.grid_rowconfigure(0, weight=1)
        self.frame3.grid_columnconfigure(0, weight=1)

        self.inspector_frame = ctk.CTkFrame(self, width=150, fg_color="gray20")
        self.inspector_frame.grid(row=1, column=1, sticky="nsew",padx=1, pady=1)
        self.inspector_frame.grid_columnconfigure(0, weight=1)

        self.tool_panel = ctk.CTkFrame(self,  fg_color="gray20")
        self.tool_panel.grid(row=1, column=0, sticky="nsew", rowspan=3,padx=1, pady=1)
        self.tool_panel.grid_rowconfigure(0, weight=1)
        self.tool_panel.grid_columnconfigure(0, weight=1)

        self.filebar = ctk.CTkFrame(self, height=40, fg_color="gray30")
        self.filebar.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=1, pady=1)
        self.filebar.grid_columnconfigure(5, weight=1)

        self.offset_frame = ctk.CTkFrame(self.frame3,fg_color="gray25")
        self.offset_frame.grid(row=0, column=0, sticky="nsew")
        self.offset_frame.grid_rowconfigure(0, weight=0)
        self.offset_frame.grid_rowconfigure(1, weight=1)
        for c in range(7):
            self.offset_frame.grid_columnconfigure(c, weight=1)

        self.properties_frame = ctk.CTkFrame(self.tool_panel, fg_color="gray20")
        self.properties_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.properties_frame.grid_columnconfigure(0, weight=1)


        # NOTE SET PROPAGATIONS
        self.frame1.grid_propagate(False)
        self.frame2.grid_propagate(False)
        self.frame3.grid_propagate(False)
        self.tool_panel.grid_propagate(False)
        self.inspector_frame.grid_propagate(False)
        self.filebar.grid_propagate(False)
        self.offset_frame.grid_propagate(False)


        # NOTE PANEL TITLE LABLES
        self.editor_lbl = ctk.CTkLabel(self.frame1, text="Editor", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.editor_lbl.grid(row=2, column=1, padx=5, pady=5, sticky="nw")

        self.visualiser_lbl = ctk.CTkLabel(self.frame2, text="Preview", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.visualiser_lbl.grid(row=0, column=0, padx=5, pady=5, sticky="nw")

        self.leftmenu_lbl = ctk.CTkLabel(self.offset_frame, text="LeftMenu", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.leftmenu_lbl.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.rightmenu_lbl = ctk.CTkLabel(self.offset_frame, text="RightMenu", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.rightmenu_lbl.grid(row=1, column=5, padx=5, pady=5, sticky="nsew")

        self.layer_lbl = ctk.CTkLabel(self.inspector_frame, text="Layers", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.layer_lbl.grid(row=0, column=0, padx=0, pady=0, sticky="new")

        self.properties_lbl = ctk.CTkLabel(self.properties_frame, text="Properties", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.properties_lbl.grid(row=0, column=0, padx=0, pady=0, sticky="new")


        # NOTE OFFSET EDITOR PANEL
        # BG IMAGE LEFT
        lcdLimg = Image.open(res("lcdFrontLeft.png")).convert("RGBA")
        lcdLimg = lcdLimg.resize((240, 240), Image.Resampling.LANCZOS)
        self.photolcdL = ctk.CTkImage(light_image=lcdLimg, dark_image=lcdLimg, size=(200, 200))

        # BG IMAGE RIGHT
        lcdRimg = Image.open(res("lcdFrontRight.png")).convert("RGBA")
        lcdRimg = lcdRimg.resize((240, 240), Image.Resampling.LANCZOS)
        self.photolcdR = ctk.CTkImage(light_image=lcdRimg, dark_image=lcdRimg, size=(200, 200))

        # OFFSET PANELS
        self.offset_visualiser_canvasL = ctk.CTkLabel(self.offset_frame, image=self.photolcdL, text="")
        self.offset_visualiser_canvasL.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.offset_visualiser_lbl1 = ctk.CTkLabel(self.offset_frame, text="Left Offsets", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.offset_visualiser_lbl1.grid(row=0, column=0, padx=5, pady=5, columnspan=3, sticky="nsew")

        self.offset_visualiser_canvasR = ctk.CTkLabel(self.offset_frame,image=self.photolcdR, text="")
        self.offset_visualiser_canvasR.grid(row=1, column=4, padx=5, pady=5, sticky="nsew")

        self.offset_visualiser_lbl2 = ctk.CTkLabel(self.offset_frame, text="Right Offsets", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.offset_visualiser_lbl2.grid(row=0, column=4, padx=5, pady=5, columnspan=3, sticky="nsew")


# ______________________________________________________________________ NOTE EDITOR AND PREVIEW PANELS ______________________________________________________________________ #

        # NOTE EDITOR CANVAS
        self.canvasLabel = tk.Label(self.frame1, bg="black")
        self.canvasLabel.pack(expand=True)

        # NOTE PREVIEW CANVAS
        self.label1 = tk.Label(self.frame2, bg="#2b2b2b")
        self.label1.pack(expand=True)

        self.canvas = tk.Canvas(self.label1, width=480, height=480, highlightthickness=0, bg="#2b2b2b", bd=0)
        self.canvas.pack(expand=True)
        self.anim_id = self.canvas.create_image(147, 242, anchor="center")
        self.anim_id2 = self.canvas.create_image(330, 242, anchor="center")
        img = Image.open(res("demoMask.png")).convert("RGBA")
        img = img.resize((480, 480), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_id = self.canvas.create_image( 240, 240, image=self.photo, anchor="center",tags="png_layer")



# ______________________________________________________________________  NOTE TOP TOOL BAR ______________________________________________________________________ #

        file_btn = ctk.CTkButton(self.filebar, text="New", height=30)
        file_btn.grid(row=0, column=0,  sticky="ew", padx=5, pady=5)

        save_btn = ctk.CTkButton(self.filebar, text="Save", command=self.saveScene, height=30)
        save_btn.grid(row=0, column=1,  sticky="ew", padx=5, pady=5)

        open_button = ctk.CTkButton(self.filebar, text="Open", command= self.loadScene, height=30)
        open_button.grid(row=0, column=2,  sticky="ew", padx=5, pady=5)

        about_btn = ctk.CTkButton(self.filebar, text="About", height=30)
        about_btn.grid(row=0, column=3,  sticky="ew", padx=5, pady=5)

        # NOTE AUTO BLINK ANIMATION TOGGLE
        self.autoBlink_switch = ctk.CTkSwitch(self.filebar, text="Auto Play", onvalue=True, offvalue=False, command=self.reset_circle)
        self.autoBlink_switch.select()
        self.autoBlink_switch.grid(row=0, column=6, padx=5, pady=5, sticky="e")

        blink_btn = ctk.CTkButton(self.filebar, text="Blink", command=self.reset_circle, height=30)
        blink_btn.grid(row=0, column=8, sticky="e", padx=5, pady=5)

        # NOTE BLINK EXPRESSION DROP DOWN MENU
        option = ctk.CTkOptionMenu(self.filebar, values=["Idle", "Happy", "Sad", "Angry", "Confused"], command=self.option_callback)
        option.grid(row=0, column=7, sticky="e", padx=5, pady=5)



# ______________________________________________________________________  NOTE PROPERTIES PANEL ______________________________________________________________________ #

        # COLOR PICKER (Embedded Widget)
        self.colorpicker = CTkColorPicker(self.properties_frame, orientation="horizontal", width=280)
        self.colorpicker.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")


        # TURN RING ON OR OFF
        self.enable_ring_switch = ctk.CTkSwitch(self.properties_frame, text="Enable Ring", onvalue=True, offvalue=False, command=self.toggle_ring )
        self.enable_ring_switch.select()   # Ring starts enabled
        self.enable_ring_switch.grid(row=5, column=0, padx=5, pady=5, sticky="we")


        # TURN RING RAINBOW ON OR OFF
        self.rainbow_ring_switch = ctk.CTkSwitch(self.properties_frame, text="Rainbow Ring", onvalue=True, offvalue=False, command=self.toggle_ring_rainbow )
        self.rainbow_ring_switch.select()
        self.rainbow_ring_switch.grid(row=6, column=0, padx=5, pady=5, sticky="we")


        # TURN IRIS ON OR OFF
        self.enable_iris_switch = ctk.CTkSwitch(self.properties_frame, text="Enable Iris", onvalue=True, offvalue=False, command=self.toggle_iris )
        self.enable_iris_switch.select()   # Ring starts enabled
        self.enable_iris_switch.grid(row=7, column=0, padx=5, pady=5, sticky="we")


        # TURN IRIS RAINBOW ON OR OFF
        self.iris_rainbow_switch = ctk.CTkSwitch(self.properties_frame, text="Rainbow Iris", onvalue=True, offvalue=False, command=self.toggle_iris_rainbow)
        self.iris_rainbow_switch.select()
        self.iris_rainbow_switch.grid(row=8, column=0, padx=5, pady=5, sticky="we")


        # TURN PUPIL ON OR OFF
        self.enable_pupil_switch = ctk.CTkSwitch( self.properties_frame, text="Enable Pupil", onvalue=True, offvalue=False, command=self.toggle_pupil)
        self.enable_pupil_switch.select()
        self.enable_pupil_switch.grid(row=9, column=0, padx=5, pady=5, sticky="we")


        # TURN PUPIL RAINBOW ON OR OFF
        self.pupil_rainbow_switch = ctk.CTkSwitch(self.properties_frame, text="Rainbow Pupil", onvalue=True, offvalue=False, command=self.toggle_pupil_rainbow )
        self.pupil_rainbow_switch.select()
        self.pupil_rainbow_switch.grid(row=10, column=0, padx=5, pady=5, sticky="we")


        # SET COLOR BUTTONS # NOTE OLD
        outterRing_btn = ctk.CTkButton( self.properties_frame,text="Outer Ring", command=lambda: [ self.setOuterRingColor(), self.turn_off_rainbow()], height=30)
        outterRing_btn.grid(row=11, column=0, padx=5, pady=5, sticky="we")

        iris_btn = ctk.CTkButton(self.properties_frame, text="Iris Color", command=lambda: [ self.setIrisColor(), self.turn_off_iris_rainbow()], height=30)
        iris_btn.grid(row=12, column=0, padx=5, pady=5, sticky="we")

        pupil_btn = ctk.CTkButton(self.properties_frame, text="Pupil Color", command= lambda: [ self.setPupilColor(), self.turn_off_pupil_rainbow()], height=30)
        pupil_btn.grid(row=13, column=0, padx=5, pady=5, sticky="we")

        clear_btn = ctk.CTkButton(self.properties_frame, text="Clear", command=self.clearScreen, height=30)
        clear_btn.grid(row=14, column=0, padx=5, pady=5, sticky="we")


# ______________________________________________________________________  NOTE LAYER SELECT ______________________________________________________________________ #

        layer1_btn = ctk.CTkButton(self.inspector_frame, text="Background", command=self.clearScreen, height=140)
        layer1_btn.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        layer2_btn = ctk.CTkButton(self.inspector_frame, text="Outter Ring", command=self.clearScreen, height=140)
        layer2_btn.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

        layer3_btn = ctk.CTkButton(self.inspector_frame, text="Iris", command=self.clearScreen, height=140)
        layer3_btn.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

        layer4_btn = ctk.CTkButton(self.inspector_frame, text="Pupil", command=self.clearScreen, height=140)
        layer4_btn.grid(row=5, column=0, padx=5, pady=5, sticky="nsew")

# ______________________________________________________________________  NOTE JSON VIEWER PANEL ______________________________________________________________________ #

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            bordercolor="#555555",
            lightcolor="#555555",
            darkcolor="#111111",
            relief="solid",
            borderwidth=1,
            rowheight=28
        )

        style.configure(
            "Treeview.Heading",
            background="#202020",
            foreground="white",
            bordercolor="#555555",
            lightcolor="#555555",
            darkcolor="#111111",
            relief="solid",
            borderwidth=1
        )

        style.map(
            "Treeview",
            background=[("selected", "#1f6aa5")],
            foreground=[("selected", "white")]
        )

        self.jsonFrame = ctk.CTkFrame(self,fg_color="gray25")
        self.jsonFrame.grid(row=2, column=1,  sticky="nsew", rowspan=2, columnspan = 2,padx=1, pady=1)
        self.jsonFrame.grid_rowconfigure(1, weight=1)
        self.jsonFrame.grid_columnconfigure(0, weight=1)

        # NOTE EXPRESSION SELECT LIST
        self.tree = ttk.Treeview(self.jsonFrame, columns=("Values"), show="tree headings")
        self.tree.heading("#0", text="Key / Index")
        self.tree.heading("Values", text="Value")
        self.tree.column("#0", width=300)
        self.tree.column("Values", width=300)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)




# ______________________________________________________________________  NOTE COMPLETION OF INIT ______________________________________________________________________ #

        # NOTE SHOW DEFAULT VALUES ON STARTUP
        self.update_json_viewer()

        # NOTE START DISPLAY ANIMATIONS
        self.toggle_animation()

        # NOTE REFRESH EDITOR AND PREVIEW
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)

        # NOTE FOR PERFORMANCE DEBUGGING
        print("UI Loaded:", time.time()-start)



# ______________________________________________________________________  NOTE FUNCTIONS ______________________________________________________________________ #


    # NOTE FOR TESTING PURPOSES
    def option_callback(self, choice):
        print(f"Selected: {choice}")


    def refreshVisualiser(self, rgb):
        h, w = rgb.shape[:2]
        # WARP LEFT AND RIGHT EYE FOR PREVIEW
        left = cv2.warpPerspective(
            rgb,
            self.M_left,
            (PREVIEW_W, PREVIEW_H),
            flags=cv2.INTER_NEAREST
        )
        right = cv2.warpPerspective(
            rgb,
            self.M_right,
            (PREVIEW_W, PREVIEW_H),
            flags=cv2.INTER_NEAREST
        )

        # UPDATE PREVIEW FRAMES
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(left))
        self.right_photo = ImageTk.PhotoImage(Image.fromarray(right))

        self.canvas.itemconfig(self.anim_id, image=self.left_photo)
        self.canvas.itemconfig(self.anim_id2, image=self.right_photo)


    ## NOTE MAY NOT NEED
    # def setLayer(self, name, layer):
    #     obj = self.getObject(name)
    #     if obj:
    #         obj["layer"] = layer
    #
    #
    ## NOTE MAY NOT NEED
    # def bringToFront(self, name):
    #     highest = max(o.get("layer", 0) for o in self.scene)
    #
    #     obj = self.getObject(name)
    #     if obj:
    #         obj["layer"] = highest + 1
    #
    #
    ## NOTE MAY NOT NEED
    # def sendToBack(self, name):
    #     lowest = min(o.get("layer", 0) for o in self.scene)
    #
    #     obj = self.getObject(name)
    #     if obj:
    #         obj["layer"] = lowest - 1


    def setProperty(self, name, key, value):
        obj = self.getObject(name)

        if obj:
            obj[key] = value
            self.onSceneChanged()


    def onSceneChanged(self):
        self.update_json_viewer()
        self.renderUpdate()

    def getObject(self, name):
        return self.objects.get(name)

    def mergeSceneDefaults(self, loaded_scene):
        default_objects = {
            obj["name"]: obj
            for obj in self.default_scene
        }

        merged_scene = []

        for obj in loaded_scene:

            name = obj.get("name")

            # Start with default object
            if name in default_objects:
                merged = copy.deepcopy(default_objects[name])
            else:
                merged = {}

            # Overwrite with loaded values
            merged.update(obj)

            merged_scene.append(merged)

        return merged_scene


    def turn_off_rainbow(self):
        self.rainbow_ring_switch.deselect()
        self.getObject("ring")["rainbow"] = False


    def turn_off_iris_rainbow(self):
        self.iris_rainbow_switch.deselect()
        self.getObject("iris")["rainbow"] = False


    def turn_off_pupil_rainbow(self):
        self.pupil_rainbow_switch.deselect()
        self.getObject("pupil")["rainbow"] = False


    def toggle_iris(self):
        self.getObject("iris")["enabled"] = self.enable_iris_switch.get()


    def toggle_iris_rainbow(self):
        self.setProperty("iris", "rainbow", self.iris_rainbow_switch.get())


    def toggle_ring(self):
        self.setProperty("ring", "enabled", self.enable_ring_switch.get())


    def toggle_ring_rainbow(self):
        self.setProperty("ring", "rainbow", self.rainbow_ring_switch.get())


    def toggle_pupil(self):
        self.getObject("pupil")["enabled"] = self.enable_pupil_switch.get()


    def toggle_pupil_rainbow(self):
        self.setProperty("pupil", "rainbow", self.pupil_rainbow_switch.get())


    def setOuterRingColor(self):
        ring = self.getObject("ring")
        self.setProperty("ring", "color", list(self.hex_to_bgr(self.colorpicker.get())))
        ring["rainbow"] = False
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)

    def setIrisColor(self):
        iris = self.getObject("iris")
        self.setProperty("iris", "color", list(self.hex_to_bgr(self.colorpicker.get())))
        iris["rainbow"] = False
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)


    def setPupilColor(self):
        pupil = self.getObject("pupil")
        self.setProperty("pupil", "color", list(self.hex_to_bgr(self.colorpicker.get())))
        pupil["rainbow"] = False
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)


    def refreshImage(self, rgb):
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        self.canvasLabel.configure(image=imgtk)
        self.canvasLabel.image = imgtk


    def hex_to_bgr(self, colour):
        colour = colour.lstrip('#')
        r = int(colour[0:2], 16)
        g = int(colour[2:4], 16)
        b = int(colour[4:6], 16)
        return (b, g, r)


    def reset_circle(self):
        self.y_pos = 0
        self.going_down = True
        self.blinking = True


    def toggle_animation(self):
        """Start/stop animation"""
        if not self.running:
            self.running = True
            self.show_frame()
        else:
            self.running = False


    def drawSpiral(self, frame, obj):

        angle = math.radians(obj["rotation"])

        c = math.cos(angle)
        s = math.sin(angle)

        rot = np.array([
            [c, -s],
            [s,  c]
        ], dtype=np.float32)

        pts = self.spiral_points @ rot.T

        pts[:, 0] += obj["center"][0]
        pts[:, 1] += obj["center"][1]

        cv2.polylines(
            frame,
            [pts.astype(np.int32)],
            False,
            obj["color"],
            obj["thickness"],
            cv2.LINE_AA
        )


    def renderScene(self, frame):

        for obj in self.scene:

            if not obj["enabled"]:
                continue

            draw_obj = obj.copy()

            if obj["name"] == "spiral":
                draw_obj["rotation"] = self.anim_rotation

            if obj["name"] == "eyelid":
                draw_obj["center"] = [
                    obj["center"][0],
                    self.anim_eyelid_y
                ]


            if draw_obj["type"] == "ellipse":
                thickness = -1 if draw_obj["fill"] else 2

                cv2.ellipse(
                    frame,
                    tuple(draw_obj["center"]),
                    tuple(draw_obj["axes"]),
                    0,
                    0,
                    360,
                    tuple(draw_obj["color"]),
                    thickness
                )

            elif draw_obj["type"] == "circle":

                thickness = -1 if draw_obj["fill"] else 2

                cv2.circle(
                    frame,
                    tuple(draw_obj["center"]),
                    draw_obj["radius"],
                    tuple(draw_obj["color"]),
                    thickness
                )

            elif draw_obj["type"] == "ring":

                cv2.circle(
                    frame,
                    tuple(draw_obj["center"]),
                    draw_obj["radius"],
                    tuple(draw_obj["color"]),
                    draw_obj["thickness"]
                )

            elif draw_obj["type"] == "spiral":
                self.drawSpiral(frame, draw_obj)


    def updateColours(self):
        self.hue_cycle = (self.hue_cycle + 2) % 360

        r,g,b = colorsys.hsv_to_rgb(self.hue_cycle/360, 1, 1)

        colour = [int(r*255), int(g*255), int(b*255)]

        for obj in self.scene:
            if obj.get("rainbow", False):
                obj["color"] = colour


    def updatePhysics(self):
        if not self.blinking:
            return

        speed = 18
        max_blink = HEIGHT // 2

        if self.going_down:

            self.y_pos += speed

            if self.y_pos >= max_blink:
                self.y_pos = max_blink
                self.going_down = False

        else:

            self.y_pos -= speed

            if self.y_pos <= 0:
                self.y_pos = 0
                self.blinking = False


    def updateAnimation(self):
        self.angle = (self.angle + 5) % 360
        self.anim_rotation = self.angle
        self.anim_eyelid_y = self.y_pos

    def renderUpdate(self):
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)

    def show_frame(self):
        if not self.running:
            return
        self.updatePhysics()
        self.updateColours()
        self.updateAnimation()
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)
        self.canvas.after(30,self.show_frame)


    def clearScreen(self):
        self.scene = copy.deepcopy(self.default_scene)
        self.objects = {
            obj["name"]: obj
            for obj in self.scene
        }
        self.y_pos = 0
        self.hue_cycle = 0
        self.angle = 0
        self.blinking = False
        self.going_down = True
        self.updateControls()
        self.image.fill(0)
        self.renderScene(self.image)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)


    def saveScene(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="eye.json"
        )
        if not filename:
            return

        with open(filename, "w") as f:
            json.dump(self.scene, f, indent=4)


    # SYNCHRONIZES APP WITH FILE WWHEN "loadScene" CALLED
    def updateControls(self):
        # Ring
        ring = self.getObject("ring")
        if ring["enabled"]:
            self.enable_ring_switch.select()
        else:
            self.enable_ring_switch.deselect()

        if ring.get("rainbow", False):
            self.rainbow_ring_switch.select()
        else:
            self.rainbow_ring_switch.deselect()

        # Iris
        iris = self.getObject("iris")
        if iris["enabled"]:
            self.enable_iris_switch.select()
        else:
            self.enable_iris_switch.deselect()

        if iris.get("rainbow", False):
            self.iris_rainbow_switch.select()
        else:
            self.iris_rainbow_switch.deselect()

        # Pupil
        pupil = self.getObject("pupil")
        if pupil["enabled"]:
            self.enable_pupil_switch.select()
        else:
            self.enable_pupil_switch.deselect()

        if pupil.get("rainbow", False):
            self.pupil_rainbow_switch.select()
        else:
            self.pupil_rainbow_switch.deselect()


    # Function to recursively display JSON data
    def display_json(self, data, parent=""):

        if isinstance(data, dict):

            for key, value in data.items():

                if isinstance(value, (dict, list)):
                    node = self.tree.insert(parent, "end", text=key, values=("",))
                    self.display_json(value, node)

                else:
                    self.tree.insert(parent, "end", text=key, values=(value,))

        elif isinstance(data, list):

            for index, value in enumerate(data):

                if isinstance(value, dict) and "name" in value:
                    label = f"[{index}] {value['name']}"
                else:
                    label = f"[{index}]"

                if isinstance(value, (dict, list)):
                    node = self.tree.insert(parent, "end", text=label, values=("",))
                    self.display_json(value, node)

                else:
                    self.tree.insert(parent, "end", text=label, values=(value,))

    def update_json_viewer(self):
        self.tree.delete(*self.tree.get_children())
        self.display_json(self.scene)


    def loadScene(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

        if not filename:
            return

        with open(filename, "r") as f:
            loaded_scene = json.load(f)

        self.scene = self.mergeSceneDefaults(loaded_scene)

        self.tree.delete(*self.tree.get_children())
        self.display_json(self.scene, "")

        for obj in self.scene:
            obj.setdefault("enabled", True)
            obj.setdefault("layer", 0)
            obj.setdefault("rainbow", False)

        self.scene.sort(key=lambda o: o["layer"])

        self.objects = {
            obj["name"]: obj
            for obj in self.scene
        }

        self.default_scene = copy.deepcopy(self.scene)


        # NOTE REBUILD SPIRAL CACHE FROM LOADED SETTINGS
        spiral = self.getObject("spiral")

        if spiral:
            self.spiral_points = []

            for theta in np.linspace(
                0,
                spiral["turns"] * 2 * math.pi,
                250
            ):
                r = spiral["radius"] * (
                    theta / (spiral["turns"] * 2 * math.pi)
                )

                self.spiral_points.append([
                    r * math.cos(theta),
                    r * math.sin(theta)
                ])

            self.spiral_points = np.array(
                self.spiral_points,
                dtype=np.float32
            )

        self.updateControls()

        self.image.fill(0)
        self.renderScene(self.image)

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.refreshImage(rgb)
        self.refreshVisualiser(rgb)



# ______________________________________________________________________  NOTE ENTRY ______________________________________________________________________ #

if __name__ == "__main__":
    app = Application()
    app.mainloop()
