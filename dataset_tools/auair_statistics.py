#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

labels_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/labels/"

catagories_tag = ['Human', 'Car', 'Truck', 'Van', 'Motorbike', 'Bicycle', 'Bus', 'Trailer']

def get_annotation_files(path = labels_path):
    train_annotation_path = labels_path + "train"
    val_annotation_path = labels_path + "val"
    
    res = [os.path.join(train_annotation_path, files) for files in os.listdir(train_annotation_path)]
    for files in os.listdir(val_annotation_path):
        res.append(os.path.join(val_annotation_path, files))
    print("Totally %d annotation files found."%(len(res)))

    return res


def get_label_distribution_each_pic(file_list):
    res = dict()
    
    for file in file_list:
        if file in res.keys():
            print("Dataset Wrong. [%s] found twice."%file)
            sys.exit()
        
        with open(file) as f:
            fcontent = f.readlines()
        
        label_dis = [0 for i in range(8)]
        for line_content in fcontent:
            if line_content == "":
                continue
            label_index = int(line_content.split(" ")[0])
            label_dis[label_index] += 1
        
        res[file] = label_dis
    
    return res


def get_instance_per_pic(label_distribution):
    instance_dis = dict()
    
    for file, label_dis in zip(label_distribution.keys(), label_distribution.values()):
        instance_dis[file] = sum(label_dis)
    
    return instance_dis   


def get_catagories_per_pic(label_distribution):
    catagories_dis = dict()
    
    for file, label_dis in zip(label_distribution.keys(), label_distribution.values()):
        count = 0
        for item in label_dis:
            if item != 0:
                count += 1
                
        catagories_dis[file] = count
    
    return catagories_dis   


def get_instance_per_catagories(label_distribution):
    instance_per_catagories = [0 for i in range(8)]
    instance_per_catagories = np.array(instance_per_catagories)
    
    for label_dis in label_distribution.values():
        label_dis = np.array(label_dis)
        instance_per_catagories = instance_per_catagories + label_dis
    
    return list(instance_per_catagories)


def count_bar(num_list):
    min_val = min(num_list)
    max_val = max(num_list)
    y = [0 for i in range(max_val - min_val + 1)]
    x = [min_val+i for i in range(max_val - min_val + 1)]

    for item in num_list:
        y_index = item - min_val
        y[y_index] += 1

    return x,y


if __name__ == "__main__":
    file_list = get_annotation_files()
    
    label_distribution = get_label_distribution_each_pic(file_list)
    
    instance_per_pic = get_instance_per_pic(label_distribution)
    
    for instances_num, pic_name in zip(instance_per_pic.values(), instance_per_pic.keys()):
        if instances_num > 15:
            print(pic_name, instances_num)
    
    x, y = count_bar(instance_per_pic.values())
    print(y)
    y = np.array(y)
    y = y/sum(y)*100
    y_text = list(y*1.01)
    y, y_text=list(y), list(y_text)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.bar(x, y, width=0.5)
    ax.set_xlabel("instance per pic")
    ax.set_ylabel("Percent(%)")
    for idx in range(len(y)):
        ax.text(x[idx], y_text[idx], "%.2f"%y[idx], horizontalalignment = 'center')
    plt.savefig("./instace_per_pic.png")
    mean_instance = int(sum([x[idx]*y[idx]/100 for idx in range(len(x))]))
    print("There are averagely %d instances per pic."%mean_instance)

    catagories_per_pic = get_catagories_per_pic(label_distribution)
    x, y = count_bar(catagories_per_pic.values())
    print(y)
    y = np.array(y)
    y = y/sum(y)*100
    y_text = list(y*1.01)
    y, y_text=list(y), list(y_text)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.bar(x, y, width=0.5)
    ax.set_xlabel("catagories per pic")
    ax.set_ylabel("Percent(%)")
    for idx in range(len(y)):
        ax.text(x[idx], y_text[idx], "%.2f"%y[idx], horizontalalignment = 'center')
    plt.savefig("./catagories_per_pic.png")
    mean_catagory = int(sum([x[idx]*y[idx]/100 for idx in range(len(x))]))
    print("There are averagely %d catagories per pic."%mean_catagory)
    
    instance_per_catagories = get_instance_per_catagories(label_distribution)
    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    ax.bar([i+1 for i in range(8)], instance_per_catagories, width = 0.5)
    ax.set_xlabel("catagories")
    ax.set_ylabel("instances per catagory")
    plt.xticks([i+1 for i in range(8)], catagories_tag, rotation=90)
    plt.savefig("./instances_per_catagories.png")
    
    
