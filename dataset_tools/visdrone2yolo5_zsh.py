# -*- coding:utf-8 -*-


import os
import shutil
import cv2

visdrone_root_path = "/home/shawn/PyVenv/VedioObjectDetection/data/VisDrone/"
cp = shutil.copyfile
rmf = shutil.rmtree

def org_annotation(dir):
    files = os.listdir(dir)

    annotation_files = []
    for f in files:
        if f.endswith("txt"):
            annotation_files.append(f)

    for one_video_annoataion_file in annotation_files:

        # get img size in each video
        sequence_name = one_video_annoataion_file.split(".txt")[0]
        frame_size = {}
        for one_frame in os.listdir(dir + "../sequences/" + sequence_name + "/"):
            img = cv2.imread(dir + "../sequences/" + sequence_name + "/" + one_frame)
            img_height, img_width, _ = img.shape
            frame_index = int(one_frame.split(".jpg")[0])
            frame_size[frame_index] = [img_height, img_width]

        # read annotation file of one video
        with open(dir + one_video_annoataion_file) as f:
            fcontent = f.read()
        fcontent = fcontent.split("\n")

        # arrange new annotation file for each frame
        yolov5_annotation = {}
        for annotation_text in fcontent:
            if annotation_text == "":
                continue
            frame_index, _, bbox_left, bbox_top, bbox_width, bbox_height, _, object_category, _, _ = \
            annotation_text.split(",")

            frame_index, bbox_left, bbox_top, bbox_width, bbox_height, object_category = \
                int(frame_index), int(bbox_left), int(bbox_top), int(bbox_width), int(bbox_height), int(object_category)

            x, y = bbox_left+bbox_width/2, bbox_top+bbox_height/2
            img_height, img_width = frame_size[frame_index]
            x, y, w, h = x/img_width, y/img_height, bbox_width/img_width, bbox_height/img_height
            if frame_index not in yolov5_annotation.keys():
                yolov5_annotation[frame_index] = [[str(object_category), str(x), str(y), str(w), str(h)]]
            else:
                yolov5_annotation[frame_index].append([str(object_category), str(x), str(y), str(w), str(h)])

        for frame_index, annotations in zip(yolov5_annotation.keys(), yolov5_annotation.values()):
            frame_annotation_file_name = sequence_name + "_%d.txt"%frame_index
            f_path = dir + "../labels/" + frame_annotation_file_name
            with open(f_path,'w') as f:
                for anno in annotations:
                    f.write(" ".join(anno)+"\n")


def org_frame(dir):
    sequences = os.listdir(dir)

    for one_sequence in sequences:
        frames = os.listdir(dir + one_sequence)

        for one_frame in frames:
            frame_index = int(one_frame.split(".jpg")[0])
            frame_new_name = one_sequence + "_%d.jpg"%frame_index
            source_file = dir + one_sequence + "/" + one_frame
            dest_file = dir + "../images/" + frame_new_name
            cp(source_file, dest_file)


def visdrone2yolo5(dir):
    # refresh labels and images dir
    if os.path.exists(dir + "labels/"):
        rmf(dir + "labels/")
    os.mkdir(dir + "labels/")
    if os.path.exists(dir + "images/"):
        rmf(dir + "images/")
    os.mkdir(dir + "images/")

    # org annotation files
    org_annotation(dir + "annotations/")

    # org frames
    org_frame(dir + "sequences/")



if __name__ == "__main__":
    for dir in ['VisDrone2019-VID-train/', 'VisDrone2019-VID-val/', 'VisDrone2019-VID-test-dev/']:
    #for dir in ['VisDrone2019-VID-test-dev/']:
        visdrone2yolo5(visdrone_root_path + dir)