import auairtools.auair as aat
import os
import sys
import random as biubiubiu
import shutil

json_path = "D:\\[0]MyFiles\\PyVirtualEnv\\data\\VedioObjectDetection\\auair_dataset\\annotations.json"
data_path = "D:\\[0]MyFiles\\PyVirtualEnv\\data\\VedioObjectDetection\\auair_dataset\\images\\"
label_path = "D:\\[0]MyFiles\\PyVirtualEnv\\data\\VedioObjectDetection\\auair_dataset\\labels\\"
prepare_flag_file = "D:\\[0]MyFiles\\PyVirtualEnv\\data\\VedioObjectDetection\\auair_dataset\\prepare_finished.txt"
split_finished_flag_file = "D:\\[0]MyFiles\\PyVirtualEnv\\data\\VedioObjectDetection\\auair_dataset\\split_finished.txt"

def prepare_data_from_auair():
    if os.path.exists(prepare_flag_file):
        return

    auair_data = aat.AUAIR(json_path, data_path)
    data_num = auair_data.num_samples

    for i in range(data_num):
        res = auair_data.get_yolo_format_data_by_index(i)
        image_name = auair_data.get_image_name_by_index(i)
        image_width, image_height = auair_data.get_image_size_by_index(i)
        image_name = image_name.split(".")[0]

        with open(label_path + image_name + ".txt", 'w') as f:
            for idx, bar in enumerate(res):
                class_index, x, y, w, h = bar[0], bar[1], bar[2], bar[3], bar[4]
                x, y, w, h = x/image_width, y/image_height, w/image_width, h/image_height
                f.write(str(class_index)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                if idx != len(res) - 1:
                    f.write("\n")
        if i % 1000 == 0:
            print("%s finished."%(label_path + image_name + ".txt"))

    with open(prepare_flag_file, 'w') as f:
        f.write("")

def split_train_data(ratio):
    if not os.path.exists(prepare_flag_file):
        sys.exit("Dataset not prepared.")

    if os.path.exists(split_finished_flag_file):
        return

    data_path_list = [data_path+item for item in os.listdir(data_path) if os.path.isfile(data_path+item)]
    label_path_list = [label_path+item for item in os.listdir(label_path) if os.path.isfile(label_path+item)]
    
    if len(data_path_list) != len(label_path_list):
        sys.exit("images and labels dont equal.")

    index_list = list(range(len(data_path_list)))
    biubiubiu.shuffle(index_list)

    train_num = int(ratio * len(data_path_list))
    val_num = len(data_path_list) - train_num
    print("traing data num: %d"%train_num)
    print("validating data num: %d"%val_num)

    train_index = index_list[0:train_num]
    val_index = index_list[train_num:]

    train_inp = [data_path_list[idx] for idx in train_index]
    train_label = [label_path_list[idx] for idx in train_index]

    val_inp = [data_path_list[idx] for idx in val_index]
    val_label = [label_path_list[idx] for idx in val_index]

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

    for i in range(val_num):
        shutil.move(val_inp[i], data_path+"val")
        shutil.move(val_label[i], label_path+"val")

    with open(split_finished_flag_file, 'w') as f:
        f.write("")

    

if __name__ == "__main__":
    prepare_data_from_auair()
    split_train_data(0.8)