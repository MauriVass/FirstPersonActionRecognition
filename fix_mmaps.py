import os
import shutil
FRAMES_PATH = os.path.join("GTEA61", "processed_frames2")


def extract_id(filename):
    return int(filename.replace("map", "").replace("rgb", "").replace(".png", ""))


def fix_mmaps(path):
    for user_dir in os.listdir(path):
        user_path = os.path.join(path, user_dir)
        for action in os.listdir(user_path):
            action_path = os.path.join(user_path, action)
            for instance in os.listdir(action_path):
                rgb_frames = sorted(os.listdir(os.path.join(action_path, instance, "rgb")))
                map_frames = sorted(os.listdir(os.path.join(action_path, instance, "mmaps")))
                if len(rgb_frames) != len(map_frames):
                    map_ids = [extract_id(filename) for filename in map_frames]
                    rgb_ids = [extract_id(filename) for filename in rgb_frames]
                    missing_maps_ids = [id for id in rgb_ids if id not in map_ids]
                    for map_id in missing_maps_ids:
                        next_id = map_id + 1
                        while next_id in missing_maps_ids:
                            next_id += 1
                        shutil.copy(os.path.join(action_path, instance, "mmaps", f"map{str(next_id).zfill(4)}.png"),
                                    os.path.join(action_path, instance, "mmaps", f"map{str(map_id).zfill(4)}.png"))

if __name__ == '__main__':
    fix_mmaps(FRAMES_PATH)