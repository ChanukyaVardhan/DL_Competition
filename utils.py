import numpy as np
import torch
import random

shapes = ["cube", "sphere", "cylinder"]
materials = ["metal", "rubber"]
colors = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]


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

# Same as above but returns an in in range [0, 48]
def get_unique_objects_id(masks):
    return np.unique(masks).tolist()

def apply_background_heuristic(S, uniq):
    random.seed(3)  # our team number
    CNT = 0
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

    print("Videos that need fixing : ", CNT)
    return S


def get_area_and_neighbours(msk, k, known_ids):
    area = 0
    H, W = msk.shape
    neighbours = set()
    # visited = [[False] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):
            if msk[i][j] == k:
                area += 1
                if i > 0 and msk[i-1][j] != k and msk[i-1][j] != 0:
                    neighbours.add(msk[i-1][j])
                if i < H-1 and msk[i+1][j] != k and msk[i+1][j] != 0:
                    neighbours.add(msk[i+1][j])
                if j > 0 and msk[i][j-1] != k and msk[i][j-1] != 0:
                    neighbours.add(msk[i][j-1])
                if j < W-1 and msk[i][j+1] != k and msk[i][j+1] != 0:
                    neighbours.add(msk[i][j+1])

    return area, [n for n in neighbours if n in known_ids]

# FIX : Write a better heuristic. This is just a placeholder


def get_closest_object(k, known_ids):
    new_class_data = get_class_ids(k)   # Shape, Material, Color
    # List of Shape, Material, Color
    known_class_data = [get_class_ids(i) for i in known_ids]

    # Check if there is a known object with same shape and color but different material
    priorities = []
    for i, class_data in enumerate(known_class_data):
        priority = 0
        priority += (class_data[0] != new_class_data[0]
                     ) * 2    # Different shape
        priority += (class_data[1] != new_class_data[1]
                     ) * 1    # Different material
        priority += (class_data[2] != new_class_data[2]
                     ) * 4    # Different color
        priorities.append((priority, i))

    priorities.sort()

    return known_ids[priorities[0][1]]


def apply_connected_components_heuristic(S, uniq):
    random.seed(3)  # our team number
    bad_cnt = 0
    area_threshold = 100  # FIX: Tune this!
    print("Running connected components heuristic with area threshold : ", area_threshold)
    for i, obj in enumerate(uniq):  # Iterating over batch
        msk = S[i].clone()
        msk = msk.detach().cpu().numpy()
        # print("msk shape : ", msk.shape)
        uniq_msk = np.unique(msk)
        good = True
        known_ids = [int(1 + C + 8*B + 16*A) for (A, B, C) in obj]
        obj_mapping = {}
        for k in uniq_msk:
            if k == 0:
                continue
            if k not in known_ids:  # Unknown object
                good = False
                obj_area, neighbours = get_area_and_neighbours(
                    msk, k, known_ids)
                if obj_area < area_threshold:  # Small object
                    obj_mapping[k] = 0
                    if len(neighbours) > 0:
                        # Returns the first neighbour (known object)
                        obj_mapping[k] = neighbours[0]
                else:  # Large object
                    if len(neighbours) > 0:
                        # Returns the first neighbour (known object)
                        obj_mapping[k] = neighbours[0]
                    else:
                        obj_mapping[k] = get_closest_object(k, known_ids)

            else:
                obj_area, neighbours = get_area_and_neighbours(
                    msk, k, known_ids)
                if obj_area < area_threshold:  # Small object
                    # For known objects, we assign small areas background
                    obj_mapping[k] = 0

        if not good:
            bad_cnt += 1
        for k, v in obj_mapping.items():
            S[i][S[i] == k] = v

    print("Videos that need fixing : ", bad_cnt)
    return S

# Simple heuristic: We are assigning background / known random objects to the unknown objects


def apply_heuristics(S, uniq, method):
    # S: 1000 x H x W
    # uniq: list - 1000 of numpy arrays (uniq objs over 11 frames)
    if method == "background":
        return apply_background_heuristic(S, uniq)
    elif method == "connected_components":
        return apply_connected_components_heuristic(S, uniq)
    else:
        raise Exception("Unknown heuristic method")


if __name__ == "__main__":
    print(class_labels)
