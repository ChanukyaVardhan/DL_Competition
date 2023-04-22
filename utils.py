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


class_labels = {i: get_class_name(i) for i in range(49)}

if __name__ == "__main__":
    print(class_labels)
