import traceback

from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from data_utils import objects_filter, merge_cyclist_label

def main():
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg)
    dtsave = DataSave(cfg)
    try:
        model.set_synchrony()
        model.spawn_agent()
        model.spawn_actors()
        model.set_actors_route()
        model.sensor_listen()
        step = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        USE_CYCLIST_LABEL = cfg["SAVE_CONFIG"]["USE_CYCLIST_LABEL"]
        while True:
            if step % STEP ==0:
                data = model.tick()
                data = objects_filter(data)
                if USE_CYCLIST_LABEL:
                    data = merge_cyclist_label(data)
                dtsave.save_training_files(data)
                print(step / STEP)
            else:
                model.world.tick()
            step+=1
    except Exception as e:
        traceback.print_exc()
    finally:
        model.setting_recover()


if __name__ == '__main__':
    main()
