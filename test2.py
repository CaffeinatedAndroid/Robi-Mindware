# main.py
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import NodePath
import os
from panda3d.core import Vec3
# Optional: Use ikpy to parse URDF (ensure your URDF references valid visual geometry)
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

class RobotRenderer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load environment (optional)
        self.scene = self.loader.loadModel("models/environment")
        self.scene.reparentTo(self.render)
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Define the kinematic chain using ikpy (NO mesh_filename)
        self.robot_chain = Chain.from_json_file("baxter/baxter_left_arm.json")
        self.robot_chain2 = Chain.from_json_file("baxter/baxter_right_arm.json")
        # self.robot_chain = Chain(name='arm', links=[
        #     OriginLink(),
        #     URDFLink(
        #         name="base",
        #         origin_translation=[0, 0, 1],
        #         origin_orientation=[0, 0, 0],
        #         rotation=[0, 0, 1],
        #         # Remove mesh_filename
        #     ),
        #     URDFLink(
        #         name="link1",
        #         origin_translation=[0, 0, 1],
        #         origin_orientation=[0, 0, 0],
        #         rotation=[0, 1, 0],
        #         # Remove mesh_filename
        #     ),
        #     URDFLink(
        #         name="link2",
        #         origin_translation=[0, 0, 3],
        #         origin_orientation=[0, 0, 0],
        #         rotation=[0, 0, 1],
        #         # Remove mesh_filename
        #     )
        # ])

        # Render each link in Panda3D
        self.render_robot()


    def render_robot(self):
        """Render each link of the ikpy robot chain in Panda3D."""
        parent_np = self.render
        # Define a list of mesh filenames corresponding to your links (skip OriginLink)
        link_meshes = ["models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/panda.egg"]

        link_meshes2 = ["models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg", "models/box.egg"]

        for link, mesh_path in zip(self.robot_chain.links[1:], link_meshes): # Skip OriginLink
            # Load the 3D model
            model = self.loader.loadModel(mesh_path)


            model.setScale(0.1, 0.1, 0.1)  # Adjust scale as needed
            model.flattenLight()
            model.reparentTo(parent_np)
            # Position the model at the link's origin translation
            model.setPos(*link.origin_translation)

            # Store reference for later use
            model.setTag('link_name', link.name)
            setattr(self, f"link_{link.name}", model)

            # The next link is a child of this one
            parent_np = model




    def update_robot_ik(self, target_position):
        """
    #     Calculate joint angles to reach target_position and update the 3D models.
    #     target_position: A Panda3D Point3 or list [x, y, z]
        """
    #     # 1. Solve for joint angles using ikpy
        target_3d = [target_position[0], target_position[1], target_position[2]]
        angles = self.robot_chain.inverse_kinematics(target_3d)

            # 2. Apply the angles to the Panda3D models (skip the first OriginLink)
        # The first angle (index 1) corresponds to the 'base' link.
        for i, (link, angle) in enumerate(zip(self.robot_chain.links[1:], angles[1:])):
            model = getattr(self, f"link_{link.name}")
            # For a rotation around Y axis, the angle affects the H (heading) component.
            # You may need to adjust the axis (H, P, R) based on your link's rotation parameter.
            print(f"Link {link.name}, rotation: {link.rotation}, type: {type(link.rotation)}")
            #if link.rotation == [0, 0, angle]: # Rotation around Y
            model.setHpr(0,0,angle)
            # elif link.rotation == [0, 0, 1]: # Rotation around Z
            #     model.setHpr(0, 0, angle * 180 / 3.14159)

            # Add more conditions for other rotation axes as needed.
    #
    # # Example usage in your main loop or an event:
    # # renderer = RobotRenderer()
    # # renderer.update_robot_ik([1.0, 0.0, 1.5]) # Move end-effector to (1.0, 0.0, 1.5)
    # # renderer.taskMgr.step() # Render the frame



# Run the app
app = RobotRenderer()


app.update_robot_ik([0.1, 1.0, 1.0]) # Move end-effector to (1.0, 0.0, 1.5)
app.taskMgr.step() # Render the frame

app.run()
