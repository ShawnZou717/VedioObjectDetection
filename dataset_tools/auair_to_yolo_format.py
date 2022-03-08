import auairtools.auair as aat
import os
import sys
import random as biubiubiu
import shutil

json_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/annotations.json"
data_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/images/"
label_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/labels/"
prepare_flag_file = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/prepare_finished.txt"
split_finished_flag_file = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/split_finished.txt"

data_path_origin = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_origin/images/"

def prepare_data_from_auair():
    auair_data = aat.AUAIR(json_path, data_path)

    if os.path.exists(prepare_flag_file):
        return auair_data

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    data_num = auair_data.num_samples

    for i in range(data_num):
        res = auair_data.get_yolo_format_data_by_index(i)
        image_name = auair_data.get_image_name_by_index(i)
        image_width, image_height = auair_data.get_image_size_by_index(i)
        image_name = image_name.split(".")[0]

        with open(label_path + image_name + ".txt", 'w') as f:
            for idx, bar in enumerate(res):
                class_index, x, y, w, h = bar[0], bar[1], bar[2], bar[3], bar[4]
                x, y = x+w/2, y+h/2 # yolo demands the coordinates to be center of the bbox.
                x, y, w, h = x/image_width, y/image_height, w/image_width, h/image_height
                f.write(str(class_index)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                if idx != len(res) - 1:
                    f.write("\n")
        if i % 1000 == 0:
            print("%s finished."%(label_path + image_name + ".txt"))

    with open(prepare_flag_file, 'w') as f:
        f.write("")
    return auair_data


def split_train_data(ratio, auair_data):
    if not os.path.exists(prepare_flag_file):
        sys.exit("Dataset not prepared.")

    if os.path.exists(split_finished_flag_file):
        return

    image_name_list = []
    for i in range(auair_data.num_samples):
        image_name_list.append(auair_data.get_image_name_by_index(i))

    biubiubiu.shuffle(image_name_list)

    train_num = int(ratio * auair_data.num_samples)
    val_num = auair_data.num_samples - train_num
    print("train data num: %d"%train_num)
    print("validating data num: %d"%val_num)

    train_image = image_name_list[0:train_num]
    val_image = image_name_list[train_num:]

    train_inp = [data_path + image_name for image_name in train_image]
    train_label = [label_path + image_name.split(".")[0] + ".txt" for image_name in train_image]

    val_inp = [data_path + image_name for image_name in val_image]
    val_label = [label_path + image_name.split(".")[0] + ".txt" for image_name in val_image]

    if not os.path.exists(data_path+"train"):
        os.mkdir(data_path+"train")
    if not os.path.exists(data_path+"val"):
        os.mkdir(data_path+"val")
    if not os.path.exists(label_path+"train"):
        os.mkdir(label_path+"train")
    if not os.path.exists(label_path+"val"):
        os.mkdir(label_path+"val")

    for i in range(train_num):
        shutil.move(train_inp[i], data_path+"train")
        shutil.move(train_label[i], label_path+"train")
        if i % 1000 == 0:
            print("%d th train 1000 moved."%i)

    for i in range(val_num):
        shutil.move(val_inp[i], data_path+"val")
        shutil.move(val_label[i], label_path+"val")
        if i % 1000 == 0:
            print("%d th val 1000 moved."%i)

    with open(split_finished_flag_file, 'w') as f:
        f.write("")


if __name__ == "__main__":
    auair_data = prepare_data_from_auair()
    split_train_data(0.8, auair_data)

    if len(sys.argv) == 2:
        a = aat.AUAIR(json_path, data_path_origin)
        a.display_bboxes(sys.argv[1])

