# Some necessary imports
import numpy as np
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
import cv2
from ikpy.chain import Chain
from ikpy.utils import plot
from mpl_toolkits.mplot3d import Axes3D;
# Optional: support for 3D plotting in the NB
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from ikpy.chain import Chain
# turn this off, if you don't need it

baxter_left_arm_chain = Chain.from_json_file("baxter/baxter_left_arm.json")
baxter_right_arm_chain = Chain.from_json_file("baxter/baxter_right_arm.json")
baxter_pedestal_chain = Chain.from_json_file("baxter/baxter_pedestal.json")
baxter_head_chain = Chain.from_json_file("baxter/baxter_head.json")


### Let's try some IK
ax = plt.figure().add_subplot(111, projection='3d')


target = [1, 0.5, 1]
target_orientation = [1, 0, 0]

frame_target = np.eye(4)
frame_target[:3, 3] = target

ik = baxter_left_arm_chain.inverse_kinematics_frame(frame_target)

baxter_left_arm_chain.plot(ik, ax, target=target)
baxter_right_arm_chain.plot([0] * (len(baxter_left_arm_chain)), ax)
baxter_pedestal_chain.plot([0] * (2 + 2), ax)
baxter_head_chain.plot([0] * (4 + 2), ax)
ax.legend()
plt.show()
