import traceback

from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from data_utils import objects_filter, merge_cyclist_label, merge_car_label, reverse_rotation, kitti_label_filter

def filter_empty_scene():
    import os

    folder = "data/test/training/kitti_label"
    output_file = os.path.join("data/test/training", "filtered_train.txt")

    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    non_empty_files = [f for f in txt_files if os.path.getsize(os.path.join(folder, f)) > 0]

    names = [os.path.splitext(f)[0] for f in non_empty_files]

    names.sort()

    with open(output_file, "w", encoding="utf-8") as outfile:
        for name in names:
            outfile.write(name + "\n")

    print(f"共找到 {len(names)} 个非空文件，结果已保存到：{output_file}")


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
        MERGE_CAR_LABEL = cfg["SAVE_CONFIG"]["MERGE_CAR_LABEL"]
        KITTI_LABEL = cfg["KITTI_CONFIG"]["LABELS"]
        while True:
            if step % STEP ==0:
                data = model.tick()
                data = objects_filter(data)
                if USE_CYCLIST_LABEL:
                    data = merge_cyclist_label(data)
                
                if MERGE_CAR_LABEL:
                    data = merge_car_label(data)
                
                if KITTI_LABEL is not None:
                    data = kitti_label_filter(data, KITTI_LABEL)

                data = reverse_rotation(data)
                dtsave.save_training_files(data)
                print(step / STEP)
            else:
                model.world.tick()
            step+=1
    except Exception as e:
        traceback.print_exc()
    finally:
        model.setting_recover()

    filter_empty_scene()

if __name__ == '__main__':
    main()
