import carla
import time
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())
from recorder.actor_tree import ActorTree
from param import *

def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(2.0)
    carla_world = carla_client.get_world()
    carla_client.reload_world()
    actor_tree = ActorTree(carla_world,
                           "{}/config/actor_settings_template.json".format(ROOT_PATH),
                           "/tmp")
    actor_tree.init()
    actor_tree.print_tree()
    time.sleep(3)
    actor_tree.destroy()


if __name__ == "__main__":
    # execute only if run as a script
    main()