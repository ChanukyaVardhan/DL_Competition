import os
import torch
from utils import get_unique_objects_id
import numpy as np
import tqdm

def is_clean_video(mask):
    first_11_frame_objs = get_unique_objects_id(mask[:11])
    last_11_frame_objs = get_unique_objects_id(mask[11:22])
    for obj in last_11_frame_objs:
        if obj not in first_11_frame_objs:
            return False
    return True

if __name__ == "__main__":
    video_path = "data/unlabeled"
    videos = os.listdir(video_path)
    # print(videos)
    clean_videos = []
    tc = 0
    for video in tqdm(videos):
        tc += 1
        mask_path = os.path.join(video_path, video, "mask.npy")
        mask = np.load(mask_path)
        if (is_clean_video(mask)):
            clean_videos.append(os.path.join(video_path, video))
    
    print("clean video count : ", len(clean_videos), "total videos : ", tc)
    with open ("clean_videos_train.txt", "w") as f:
        for video in clean_videos:
            f.write(video + "\n")