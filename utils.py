import numpy as np
import torch
import random

shapes = ["cube", "sph", "cyl"]
materials = ["mtl", "rbr"]
colors = ["gry", "red", "blu", "grn", "brwn", "cyan", "prple", "yllw"]


def get_id(the_object):
    color = the_object['color']
    material = the_object['material']
    shape = the_object['shape']

    c_id = colors.index(color)
    m_id = materials.index(material)
    s_id = shapes.index(shape)

    obj_id = s_id * 16 + m_id * 8 + c_id + 1

    return obj_id


def get_class_name(id):
    if id == 0:
        return "background"
    else:
        id -= 1
        s_id = id // 16
        id = id % 16
        m_id = id // 8
        id = id % 8
        c_id = id

        return f"{shapes[s_id]}_{materials[m_id]}_{colors[c_id]}"


def get_class_ids(id):
    if id == 0:
        raise Exception("Background has no class id tuple")
    else:
        id -= 1
        s_id = id // 16
        id = id % 16
        m_id = id // 8
        id = id % 8
        c_id = id

        return s_id, m_id, c_id


class_labels = {i: get_class_name(i) for i in range(49)}
class_labels_list = [get_class_name(i) for i in range(49)]


def get_unique_objects(masks):
    B, T, H, W = masks.shape
    # print(B, T, H, W)
    unique_objects = []
    for b in range(B):
        per_image_unique_objects = np.array([])
        for t in range(T):
            uniq = np.unique(masks[b, t])
            per_image_unique_objects = np.union1d(
                per_image_unique_objects, uniq)
        obj_classes = [get_class_ids(i)
                       for i in per_image_unique_objects if i != 0]
        unique_objects.append(obj_classes)

    return unique_objects


# Simple heuristic: We are assigning background / known random objects to the unknown objects
def apply_heuristics(S, uniq):
    # S: 1000 x H x W
    # uniq: list - 1000 of numpy arrays (uniq objs over 11 frames)
    random.seed(3)  # our team number
    CNT = 0
    better_stack = []
    for i, obj in enumerate(uniq):
        msk = S[i].clone()
        msk = msk.detach().cpu().numpy()
        uniq_msk = np.unique(msk)
        good = True
        known_ids = [int(1 + C + 8*B + 16*A) for (A, B, C) in obj]
        obj_mapping = {}
        for k in uniq_msk:
            if k != 0:
                if k not in known_ids:
                    good = False
                    rnd = random.randrange(len(known_ids) + 10)
                    # obj_mapping[k] = (known_ids[rnd] if rnd < len(known_ids) else 0)
                    obj_mapping[k] = 0
        if not good:
            CNT += 1
        for k, v in obj_mapping.items():
            S[i][S[i] == k] = v

    print("Images that need fixing : ", CNT)
    return S


if __name__ == "__main__":
    print(class_labels)
