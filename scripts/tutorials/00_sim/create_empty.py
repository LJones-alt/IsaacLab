# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Test Environment Launch .")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim
import torch 
import isaaclab.utils.math as isaacMath
import isaacsimcore.utils.prims as prim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg


def design_env():
    #Ground Plane
    groundplane = sim.GroundPlaneCfg()
    groundplane.func("/World/defaultGroundPlane", groundplane)
    #Lighting
    distantlight = sim.DiskLightCfg(intensity=3000.0, color=(0.75,0.75,0.75))
    distantlight.func("/World/lightDistant", distantlight, translation=(1,0,10))
    #make an object - cube 
    cube_obj = sim.CuboidCfg(size=(1,1,1), visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
    cube_obj.func("/World/Object", cube_obj, translation=(0.0, 0.0, 0.5))
    #make a cone object
    origin = [[0.25, 0.25, 0.0]]
    prim_utils.create_prim(f"/World/Origin{0}","Xform", translation=origin)
    cone_obj = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim.ConeCfg(radius=0.1, height=0.2,
                          rigid_props=sim.RigidBodyPropertiesCfg(),
                          mass_props=sim.MassPropertiesCfg(mass=1.0),
                          collision_props=sim.CollisionPropertiesCfg(),
                          visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0))
                          ),
        init_state=RigidObjectCfg.InitialStateCfg()
    )
    cone=RigidObject(cfg=cone_obj)
    scene_contents = {"cone":cone}
    return scene_contents , origin

def run_sim(sim_c:sim.SimulationContext, entities:dict[str,RigidObject], origin:torch.Tensor):
    ##run the simulation loop
    #setup the sim physics
    cone = entities["cone"]
    sim_dt = sim_c.get_physics_dt()
    sim_time = 0.0
    count = 0
    #Do physics 
    while simulation_app.is_running():
        if count % 250==0:
            ## then reset
            sim_time = 0.0
            count = 0
            root_state= cone.data.default_root_state.clone()
            root_state[:,:3] += origin
            root_state += isaacMath.sample_cylinder(radius=0.1, h_range=(0.25,0.5), size=cone.num_instances, device=cone.device)
            cone.write_root_pose_to_sim(root_state[:,:7])
            cone.write_root_velocity_to_sim(root_state[:,:7])
            cone.reset()
            print ("---------------------------------")
            print("[INFO] Resetting Object.....")
        cone.write_data_to_sim()
        sim_c.step()
        sim_time += sim_dt
        count+=1
        cone.update(sim_dt)
        if count%50==0:
            print(f"Root Position : {cone.data.root_state_w[:,:3]}")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    eye_loc = (2.5,2.5,2.5)
    camera_loc = (0.0, 0.0, 0.0)
    sim.set_camera_view(eye_loc, camera_loc)

    scene_entities, scene_origins = design_env()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    run_sim(sim,scene_entities, scene_origins)
    # Simulate physics
    


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
