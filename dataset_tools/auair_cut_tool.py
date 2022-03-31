#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import cv2
import auair_statistics as aas

dataset_origin_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/"
dataset_path = "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset_cut/"
labels_path = dataset_path + "labels/"
images_path = dataset_path + "images/"


def get_attribs():
    annotation_file_list = aas.get_annotation_files(dataset_origin_path + "labels/")
    image_file_list = aas.get_image_files(annotation_file_list)
    label_distribution = aas.get_label_distribution_each_pic(annotation_file_list)
    instance_per_pic = aas.get_instance_per_pic(label_distribution)
    return image_file_list, annotation_file_list, instance_per_pic


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    iou = area / min(s1, s2)
    return iou


def cal_all_iou(annotation_list):
    length = len(annotation_list)
    box_array = [[None for ii in range(length)] for i in range(length)]

    box_index = []
    for annotation in annotation_list:
        label, x, y, w, h = annotation
        xmin = x - w/2
        xmax = x + w/2
        ymin = y - h/2
        ymax = y + h/2
        box_index.append([xmin, ymin, xmax, ymax])

    for i in range(length):
        for ii in range(i+1, length):
            box1 = box_index[i]
            box2 = box_index[ii]
            box_array[i][ii] = cal_iou(box1, box2)
            box_array[ii][i] = box_array[i][ii]

    return box_array


def DFS(graph, start):
	# 创建一个set记录点是否已被遍历
    visited = set()
    # python没有直接实现栈，这里使用list模拟栈操作
    # 入栈就是向列表中append一个元素，出栈就是pop列表中最后一个元素
    stack = [[start, 0]]
    while stack:
        v, next_idx = stack[-1]
        # 临界条件：图中点没有下一个邻接点或者邻接点全部遍历完毕
        if (v not in graph) or (next_idx >=len(graph[v])):
            stack.pop()
            continue
        next = graph[v][next_idx]
        # 记录当前节点的邻接点入栈数量
        stack[-1][1] += 1
        if next in visited:
            continue
        visited.add(next)
        stack.append([next, 0])
    visited.add(start)
    res = list(visited)
    res.sort()
    return res


def split_pic(image, annotation_list, iou_threshold = 0.1):
    image_height, image_width, _ = image.shape
    for idx, item in enumerate(annotation_list):
        label, x, y, w, h = item
        annotation_list[idx] = [label, int(x*image_width), int(y*image_height), int(w*image_width), int(h*image_height)]
    iou_array = cal_all_iou(annotation_list)

    # turn iou into graph
    graph = {}
    num_annotation = len(annotation_list)
    for idx in range(num_annotation):
        for idy in range(num_annotation):
            if iou_array[idx][idy] is not None and iou_array[idx][idy] > iou_threshold:
                if idx in graph.keys():
                    graph[idx].append(idy)
                else:
                    graph[idx] = [idy]

    cls = []
    # use DFS algorithm to see if there is path connect two obj
    pic_visited = [0 for _ in range(num_annotation)]
    while sum(pic_visited) != num_annotation:
        pic_index = 0
        for i in range(num_annotation):
            if pic_visited[i] == 0:
                pic_index = i
                break
        connected_obj = DFS(graph, pic_index)
        for obj in connected_obj:
            pic_visited[obj] = 1
        cls.append(connected_obj)

    sub_images = []
    sub_images_edge = []
    for idx_cls in range(len(cls)):
        cls_item = cls[idx_cls]
        x_min, x_max, y_min, y_max = image_width, 0, image_height, 0
        for annotation_index in cls_item:
            label, x, y, w, h = annotation_list[annotation_index]
            xmin = int(x - w / 2)
            xmax = int(x + w / 2)
            ymin = int(y - h / 2)
            ymax = int(y + h / 2)
            x_min, x_max, y_min, y_max = min(x_min, xmin), max(x_max, xmax), min(y_min, ymin), max(y_max, ymax)
        w_new = x_max - x_min
        h_new = y_max - y_min
        x_min, x_max, y_min, y_max = int(max(x_min-0.414/2*w_new, 0)), int(min(x_max+0.414/2*w_new, image_width)), \
                                     int(max(y_min-0.414/2*h_new, 0)), int(min(y_max+0.414/2*h_new, image_height))
        sub_images.append(image[y_min:y_max, x_min:x_max, :])
        sub_images_edge.append([x_min, x_max, y_min, y_max])

    sub_annotations = []
    for idx_cls in range(len(cls)):
        cls_item = cls[idx_cls]
        x_min, x_max, y_min, y_max = sub_images_edge[idx_cls]
        sub_anno_tmp = []
        for annotation_index in cls_item:
            label, x, y, w, h = annotation_list[annotation_index]
            x = (x - x_min) / (x_max - x_min)
            y = (y - y_min) / (y_max - y_min)
            w = w / (x_max - x_min)
            h = h / (y_max - y_min)
            sub_anno_tmp.append([label, x, y, w, h])
        sub_annotations.append(sub_anno_tmp)

    return sub_images, sub_annotations


def test():
    img = cv2.imread(
        "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/images/train/frame_20190905142119_x_0000937.jpg")
    with open(
            "/home/shawn/PyVenv/VedioObjectDetection/data/auair_dataset/labels/train/frame_20190905142119_x_0000937.txt") as f:
        fcontent = f.read().split("\n")

    for i in range(len(fcontent)):
        fcontent[i] = fcontent[i].split(" ")
        for j in range(len(fcontent[i])):
            fcontent[i][j] = float(fcontent[i][j])

    sub_images, sub_annotations = split_pic(img, fcontent)

    for files in os.listdir("/home/shawn/scripts_output_tmp"):
        if files.endswith("jpg"):
            os.remove(os.path.join("/home/shawn/scripts_output_tmp", files))
    count = 0
    for sub_image in sub_images:
        count += 1
        cv2.imwrite("/home/shawn/scripts_output_tmp/%dth.jpg" % count, sub_image)

    for anno in sub_annotations:
        print(anno)
        print()


def org_annotation(annotation_file):
    annotation_list = []
    with open(annotation_file) as f:
        fcontent = f.read()

    fcontent = fcontent.split("\n")
    for idx, item in enumerate(fcontent):
        fcontent[idx] =item.split(" ")
        for i in range(len(fcontent[idx])):
            if i == 0:
                fcontent[idx][i] = int(fcontent[idx][i])
            else:
                fcontent[idx][i] = float(fcontent[idx][i])
        if fcontent[idx][3] == 0 or fcontent[idx][4] == 0:
            continue
        annotation_list.append(fcontent[idx])

    return annotation_list


def annotation_write(path, annotation_list):
    with open(path, 'w') as f:
        for annotation in annotation_list:
            for i in range(len(annotation)):
                annotation[i] = str(annotation[i])
            f.write(" ".join(annotation)+"\n")


if __name__ == "__main__":
    # test()


    # clear all jpg and txt
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith("txt") or filename.endswith("jpg"):
                os.remove(os.path.join(root, filename))

    if not os.path.exists(dataset_path + "images/"):
        os.mkdir(dataset_path + "images/")
    elif not os.path.exists(dataset_path + "images/train/"):
        os.mkdir(dataset_path + "images/train/")
    elif not os.path.exists(dataset_path + "images/val/"):
        os.mkdir(dataset_path + "images/val/")

    if not os.path.exists(dataset_path + "labels/"):
        os.mkdir(dataset_path + "labels/")
    elif not os.path.exists(dataset_path + "labels/train/"):
        os.mkdir(dataset_path + "labels/train/")
    elif not os.path.exists(dataset_path + "labels/val/"):
        os.mkdir(dataset_path + "labels/val/")


    sub_image_num = 0
    image_file_list, annotation_file_list, instance_per_pic = get_attribs()

    for image_file, annotation_file in zip(image_file_list, annotation_file_list):
        img = cv2.imread(image_file)
        annotation_list = org_annotation(annotation_file)
        sub_images, sub_annotations = split_pic(img, annotation_list)

        if "train" in image_file:
            sub_image_path = images_path + "train/"
            sub_annotation_path = labels_path + "train/"
        else:
            sub_image_path = images_path + "val/"
            sub_annotation_path = labels_path + "val/"

        for image, annotation in zip(sub_images, sub_annotations):
            sub_image_num += 1
            cv2.imwrite(sub_image_path + "auair_%d.jpg"%sub_image_num, image)
            annotation_write(sub_annotation_path + "auair_%d.txt"%sub_image_num, annotation)
            if sub_image_num % 3000 == 0:
                print("Totally %d sub images generated."%sub_image_num)

