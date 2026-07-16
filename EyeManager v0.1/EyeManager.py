from CTkColorPicker import CTkColorPicker
from PIL import Image, ImageTk
from tkinter import filedialog
import customtkinter as ctk
import tkinter as tk
import numpy as np
import colorsys
import math
import json
import cv2
import os
import copy

# CONFIGURE CUSTOM-TKINTER
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# GC9A01 SIMULATOR CONSTANTS
WIDTH = 240
HEIGHT = 240
CENTER = (WIDTH // 2, HEIGHT // 2)

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()

        # MAIN APPLICATION CONFIGURATION
        self.title("Eye Manager v0.1")
        self.geometry("1200x900")

        # ANIMATION STATES
        self.y_pos = 0
        self.hue_cycle = 0
        self.angle = 0
        self.running = False
        self.going_down = True
        self.blinking = False
        self.image = np.zeros((240, 240, 3), dtype=np.uint8)



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

        # COPIES SCENE AS DEFAULT FOR "clearScreen" FUNCTION
        self.default_scene = copy.deepcopy(self.scene)



# ______________________________________________________________________ NOTE WINDOW LAYOUT ______________________________________________________________________ #

        # GRID WEIGHTS
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=3)


        # MAIN FRAMES / PANELS
        self.frame1 = ctk.CTkFrame(self,fg_color="gray25")
        self.frame2 = ctk.CTkFrame(self,fg_color="gray25")
        self.frame3 = ctk.CTkFrame(self,fg_color="gray25")
        self.tool_panel = ctk.CTkFrame(self,  fg_color="gray20")
        self.inspector_frame = ctk.CTkFrame(self,  fg_color="gray20")
        self.filebar = ctk.CTkFrame(self, height=40, fg_color="gray30")
        self.offset_frame = ctk.CTkFrame(self.frame3,fg_color="gray25")


        # SET POSITIONS
        self.frame1.grid(row=1, column=1, sticky="nsew", padx=1, pady=1)
        self.frame2.grid(row=1, column=2, sticky="nsew", padx=1, pady=1)
        self.frame3.grid(row=2, column=1, sticky="nsew", columnspan=2,padx=1, pady=1)
        self.tool_panel.grid(row=1, column=0, sticky="nsew", rowspan=3,padx=1, pady=1)
        self.inspector_frame.grid(row=1, column=3, sticky="nsew", rowspan=3,padx=1, pady=1)
        self.filebar.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=1, pady=1)
        self.offset_frame.pack(side='top', fill="both", padx=25, pady=0, expand=True, anchor= "center")


        # SET PROPAGATION
        self.frame1.grid_propagate(False)
        self.frame2.grid_propagate(False)
        self.frame3.grid_propagate(False)
        self.tool_panel.grid_propagate(False)
        self.inspector_frame.grid_propagate(False)
        self.filebar.grid_propagate(False)
        self.offset_frame.grid_propagate(True)


        # PANEL NAMES
        self.editor_lbl = ctk.CTkLabel(self.frame1, text="Editor", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.editor_lbl.grid(row=2, column=1, padx=0, pady=0, sticky="nw")
        self.visualiser_lbl = ctk.CTkLabel(self.frame2, text="Preview", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.visualiser_lbl.grid(row=0, column=0, padx=0, pady=0, sticky="nw")
        self.leftmenu_lbl = ctk.CTkLabel(self.offset_frame, text="LeftMenu", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.leftmenu_lbl.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.rightmenu_lbl = ctk.CTkLabel(self.offset_frame, text="RightMenu", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.rightmenu_lbl.grid(row=1, column=4, padx=5, pady=5, sticky="nsew")


        # OFFSET BG IMAGE LEFT
        lcdLimg = Image.open("resources/lcdFrontLeft.png").convert("RGBA")
        lcdLimg = lcdLimg.resize((240, 240), Image.Resampling.LANCZOS)
        self.photolcd = ctk.CTkImage(light_image=lcdLimg, dark_image=lcdLimg, size=(240, 240))


        # OFFSET BG IMAGE RIGHT
        lcdRimg = Image.open("resources/lcdFrontRight.png").convert("RGBA")
        lcdRimg = lcdRimg.resize((240, 240), Image.Resampling.LANCZOS)
        self.photolcdR = ctk.CTkImage(light_image=lcdRimg, dark_image=lcdRimg, size=(240, 240))


        # OFFSET PANELS
        self.offset_visualiser_canvasL = ctk.CTkLabel(self.offset_frame, image=self.photolcd, text="")
        self.offset_visualiser_canvasL.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.offset_visualiser_lbl1 = ctk.CTkLabel(self.offset_frame, text="Left Offsets", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.offset_visualiser_lbl1.grid(row=0, column=0, padx=5, pady=5, columnspan=2, sticky="nsew")

        self.offset_visualiser_canvasR = ctk.CTkLabel(self.offset_frame,image=self.photolcdR, text="")
        self.offset_visualiser_canvasR.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")

        self.offset_visualiser_lbl2 = ctk.CTkLabel(self.offset_frame, text="Right Offsets", font =("Helvetica", 25), fg_color="gray20",  width=50, height=50, corner_radius=10)
        self.offset_visualiser_lbl2.grid(row=0, column=3, padx=5, pady=5, columnspan=2, sticky="nsew")


        # EDITOR CANVAS
        self.canvasLabel = tk.Label(self.frame1, bg="black")
        self.canvasLabel.pack(expand=True)


        # PREVIEW CANVAS
        self.label1 = tk.Label(self.frame2, bg="#2b2b2b")
        self.label1.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.canvas = tk.Canvas(self.label1, width=480, height=480, highlightthickness=0)
        self.canvas.pack(side='top', fill="both", expand=True)
        self.anim_id = self.canvas.create_image(147, 242, anchor="center")
        self.anim_id2 = self.canvas.create_image(330, 242, anchor="center")
        img = Image.open("resources/demoMask.png").convert("RGBA")
        img = img.resize((480, 480), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_id = self.canvas.create_image( 240, 240, image=self.photo, anchor="center",tags="png_layer")


        # REFRESH FRAMES
        self.refreshImage()
        self.refreshVisualiser()



# ______________________________________________________________________  NOTE TOP TOOL BAR ______________________________________________________________________ #

        file_btn = ctk.CTkButton(self.filebar, text="File", height=30)
        file_btn.grid(row=0, column=0,  sticky="ew", padx=5, pady=5)

        save_btn = ctk.CTkButton(self.filebar, text="Save", command=self.saveScene, height=30)
        save_btn.grid(row=0, column=1,  sticky="ew", padx=5, pady=5)

        open_button = ctk.CTkButton(self.filebar, text="Open", command= self.loadScene, height=30)
        open_button.grid(row=0, column=2,  sticky="ew", padx=5, pady=5)

        about_btn = ctk.CTkButton(self.filebar, text="About", height=30)
        about_btn.grid(row=0, column=3,  sticky="ew", padx=5, pady=5)

        start_btn = ctk.CTkButton(self.filebar, text="Start Animation", command=self.toggle_animation, height=30)
        start_btn.grid(row=0, column=4, sticky="ew", padx=5, pady=5)

        blink_btn = ctk.CTkButton(self.filebar, text="Blink", command=self.reset_circle, height=30)
        blink_btn.grid(row=0, column=5, sticky="ew", padx=5, pady=5)



# ______________________________________________________________________  NOTE LEFT TOOL BAR ______________________________________________________________________ #

        # COLOR PICKER (Embedded Widget)
        self.colorpicker = CTkColorPicker(self.tool_panel, orientation="horizontal", width=220)
        self.colorpicker.grid(row=0, column=0, padx=5, pady=5, sticky="w")


        # TURN RING ON OR OFF
        self.enable_ring_switch = ctk.CTkSwitch(self.tool_panel, text="Enable Ring", onvalue=True, offvalue=False, command=self.toggle_ring )
        self.enable_ring_switch.select()   # Ring starts enabled
        self.enable_ring_switch.grid(row=4, column=0, padx=5, pady=5, sticky="w")


        # TURN RING RAINBOW ON OR OFF
        self.rainbow_ring_switch = ctk.CTkSwitch(self.tool_panel, text="Rainbow Ring", onvalue=True, offvalue=False, command=self.toggle_ring_rainbow )
        self.rainbow_ring_switch.select()
        self.rainbow_ring_switch.grid(row=5, column=0, padx=5, pady=5, sticky="w")


        # TURN IRIS ON OR OFF
        self.enable_iris_switch = ctk.CTkSwitch(self.tool_panel, text="Enable Iris", onvalue=True, offvalue=False, command=self.toggle_iris )
        self.enable_iris_switch.select()   # Ring starts enabled
        self.enable_iris_switch.grid(row=6, column=0, padx=5, pady=5, sticky="w")


        # TURN IRIS RAINBOW ON OR OFF
        self.iris_rainbow_switch = ctk.CTkSwitch(self.tool_panel, text="Rainbow Iris", onvalue=True, offvalue=False, command=self.toggle_iris_rainbow)
        self.iris_rainbow_switch.select()
        self.iris_rainbow_switch.grid(row=7, column=0, padx=5, pady=5, sticky="w")


        # TURN PUPIL ON OR OFF
        self.enable_pupil_switch = ctk.CTkSwitch( self.tool_panel, text="Enable Pupil", onvalue=True, offvalue=False, command=self.toggle_pupil)
        self.enable_pupil_switch.select()
        self.enable_pupil_switch.grid(row=8, column=0, padx=5, pady=5, sticky="w")


        # TURN PUPIL RAINBOW ON OR OFF
        self.pupil_rainbow_switch = ctk.CTkSwitch(self.tool_panel, text="Rainbow Pupil", onvalue=True, offvalue=False, command=self.toggle_pupil_rainbow )
        self.pupil_rainbow_switch.select()
        self.pupil_rainbow_switch.grid(row=9, column=0, padx=5, pady=5, sticky="w")


        # SET COLOR BUTTONS
        outterRing_btn = ctk.CTkButton( self.tool_panel,text="Outer Ring", command=lambda: [ self.setOuterRingColor(), self.turn_off_rainbow()], height=30)
        outterRing_btn.grid(row=10, column=0, padx=5, pady=5, sticky="w")

        iris_btn = ctk.CTkButton(self.tool_panel, text="Iris Color", command=lambda: [ self.setIrisColor(), self.turn_off_iris_rainbow()], height=30)
        iris_btn.grid(row=11, column=0, padx=5, pady=5, sticky="w")

        pupil_btn = ctk.CTkButton(self.tool_panel, text="Pupil Color", command= lambda: [ self.setPupilColor(), self.turn_off_pupil_rainbow()], height=30)
        pupil_btn.grid(row=12, column=0, padx=5, pady=5, sticky="w")

        clear_btn = ctk.CTkButton(self.tool_panel, text="Clear", command=self.clearScreen, height=30)
        clear_btn.grid(row=13, column=0, padx=5, pady=5, sticky="w")



# ______________________________________________________________________  NOTE RIGHT TOOL BAR ______________________________________________________________________ #








# ______________________________________________________________________  NOTE FUNCTIONS ______________________________________________________________________ #

    def refreshVisualiser(self):

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]

        # WARP LEFT AND RIGHT EYE FOR PREVIEW
        src = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        dst_left = np.float32([
            [-20, 0],
            [w, 20],
            [0, h-20],
            [w-20, h]
        ])

        dst_right = np.float32([
            [0, 20],
            [w+20, 0],
            [20, h],
            [w, h-20]
        ])

        M_left = cv2.getPerspectiveTransform(src, dst_left)
        M_right = cv2.getPerspectiveTransform(src, dst_right)

        left = cv2.warpPerspective(rgb, M_left, (w, h))
        right = cv2.warpPerspective(rgb, M_right, (w, h))

        # PART OF WARP FOR PREVIEW
        DISPLAY_W = 125
        DISPLAY_H = 160

        left = cv2.resize(left, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(right, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

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


    def getObject(self, name):
        for obj in sorted(self.scene, key=lambda o: o.get("layer", 0)):
            if obj["name"] == name:
                return obj

        return None


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
        self.getObject("iris")["rainbow"] = self.iris_rainbow_switch.get()


    def toggle_ring(self):
        self.setProperty("ring", "enabled", self.enable_ring_switch.get())


    def toggle_ring_rainbow(self):
        self.getObject("ring")["rainbow"] = self.rainbow_ring_switch.get()


    def toggle_pupil(self):
        self.getObject("pupil")["enabled"] = self.enable_pupil_switch.get()


    def toggle_pupil_rainbow(self):
        self.getObject("pupil")["rainbow"] = self.pupil_rainbow_switch.get()


    def setOuterRingColor(self):
        ring = self.getObject("ring")
        ring["color"] = list(self.hex_to_bgr(self.colorpicker.get()))
        ring["rainbow"] = False
        self.refreshImage()
        self.refreshVisualiser()


    def setIrisColor(self):
        iris = self.getObject("iris")
        iris["color"] = list(self.hex_to_bgr(self.colorpicker.get()))
        iris["rainbow"] = False
        self.refreshImage()
        self.refreshVisualiser()


    def setPupilColor(self):
        pupil = self.getObject("pupil")
        pupil["color"] = list(self.hex_to_bgr(self.colorpicker.get()))
        pupil["rainbow"] = False
        self.refreshImage()
        self.refreshVisualiser()


    def refreshImage(self):
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
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
        pts = []
        rotation = math.radians(obj["rotation"])
        for theta in np.linspace(0, obj["turns"] * 2 * math.pi, 250):

            r = obj["radius"] * (theta / (obj["turns"] * 2 * math.pi))
            t = theta + rotation
            x = int(obj["center"][0] + r * math.cos(t))
            y = int(obj["center"][1] + r * math.sin(t))
            pts.append((x, y))

        cv2.polylines(frame, [np.array(pts)], False, obj["color"], obj["thickness"], cv2.LINE_AA)


    def renderScene(self, frame):
        for obj in sorted(self.scene, key=lambda o: o.get("layer", 0)):

            if not obj["enabled"]:
                continue

            if obj["type"] == "ellipse":
                thickness = -1 if obj["fill"] else 2
                cv2.ellipse(
                    frame,
                    tuple(obj["center"]),
                    tuple(obj["axes"]),
                    0,
                    0,
                    360,
                    tuple(obj["color"]),
                    thickness
                )

            elif obj["type"] == "circle":
                thickness = -1 if obj["fill"] else 2
                cv2.circle(
                    frame,
                    tuple(obj["center"]),
                    obj["radius"],
                    tuple(obj["color"]),
                    thickness
                )

            elif obj["type"] == "ring":
                cv2.circle(
                    frame,
                    tuple(obj["center"]),
                    obj["radius"],
                    tuple(obj["color"]),
                    obj["thickness"]
                )

            elif obj["type"] == "spiral":
                self.drawSpiral(frame, obj)


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

        spiral = self.getObject("spiral")

        if spiral:
            spiral["rotation"] = self.angle

        # Move eyelid according to blink animation
        eyelid = self.getObject("eyelid")

        if eyelid:
            eyelid["center"][1] = self.y_pos


    def show_frame(self):
        if not self.running:
            return

        self.updatePhysics()
        self.updateColours()
        self.updateAnimation()
        frame = np.zeros_like(self.image)
        self.renderScene(frame)
        self.image = frame
        self.refreshImage()
        self.refreshVisualiser()
        self.canvas.after(30, self.show_frame)


    def clearScreen(self):
        self.scene = copy.deepcopy(self.default_scene)
        self.updateControls()
        frame = np.zeros_like(self.image)
        self.renderScene(frame)
        self.image = frame
        self.refreshImage()
        self.refreshVisualiser()


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


    def loadScene(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if not filename:
            return

        with open(filename, "r") as f:
            self.scene = json.load(f)

        self.updateControls()

        frame = np.zeros_like(self.image)
        self.renderScene(frame)
        self.image = frame
        self.refreshImage()
        self.refreshVisualiser()



# ______________________________________________________________________  NOTE ENTRY ______________________________________________________________________ #

if __name__ == "__main__":
    app = Application()
    app.mainloop()
