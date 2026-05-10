from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, Vec4
from panda3d.core import AmbientLight, Vec4
from panda3d.core import Point3
from panda3d.core import WindowProperties
import tkinter as tk
from tkinter import Button


class AppTk(ShowBase):

    def __init__(self):
        ShowBase.__init__(self, windowType='none') #NOTE START PANDA 3D WITH NO DEFAULT WINDOW
        self.startTk()

        #NOTE CREATE ROOT TK WINDOW
        self.tk = self.tkRoot
        self.tk.geometry("500x400")
        self.tk.title("Robo Mindware")
        #NOTE CREATE AND PACK LABEL FRAME
        self.label_frame = tk.LabelFrame(self.tk, text='Render View', width=420, height=340)
        self.label_frame.pack()

        #NOTE CREATE BUTTON AND PACK INTO ROOT
        button = Button(self.tk, text='Click me !', bd='5', command=self.test)
        button.pack()

        #NOTE Set properties of the Panda3D window
        props = WindowProperties()
        props.set_parent_window(self.label_frame.winfo_id())  #NOTE Display within the label frame
        props.set_origin(10, 20)  #NOTE Relative to the label frame
        props.set_size(400, 300)

        self.make_default_pipe()
        self.open_default_window(props=props)

        scene = self.loader.load_model("environment")

        scene.reparent_to(self.render)
        scene.setPos(Point3(0, 200, -50))


        cube = self.loader.loadModel("models/box")

        # Add the model to the scene graph and position it
        cube.reparentTo(self.render)
        cube.setPos(Point3(0, 1, -50))
        cube.setScale(15)

        # Create the light object and set its color (RGBA)
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))

        # Attach to scene graph
        ambientLightNP = self.render.attachNewNode(ambientLight)

        # Enable for the entire scene
        self.render.setLight(ambientLightNP)


        # Create the light object
        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor(Vec4(1.0, 1.0, 1.0, 1)) # White light
        directionalLight.setShadowCaster(True, 512, 512)

        # Attach and orient (e.g., shining from top-left)
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(0, -60, 0) # HPR rotation

        # Enable for the entire scene
        self.render.setLight(directionalLightNP)
        self.render.setShaderAuto()


    def test(self):
        print("Hello")


app = AppTk()
app.run()
