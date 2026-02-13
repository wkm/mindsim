"""Launch MuJoCo viewer for a scene. Usage: python view.py <scene_path>"""

import sys

import mujoco
import mujoco.viewer


def main():
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "bots/simple2wheeler/scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
