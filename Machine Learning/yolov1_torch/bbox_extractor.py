import argparse
import xml.etree.ElementTree as ET
import os
import pickle
parser = argparse.ArgumentParser(description='Build Annotations.')
parser.add_argument('dir', default='..', help='Annotations.')

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}

classes_num2 = {v: k for k, v in classes_num.items()}

def convert_annotation(year, image_id, f,imgset):
    if imgset=="val":
        imgset="train"
    in_file = os.path.join('/media/amir/4C8C65E67913301E/Datasets/VOCdevkit/VOC%s/%s/Annotations/%s.xml' % (year, imgset,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    boxes=[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        classes = list(classes_num.keys())
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text),cls_id)
        #f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))
        boxes.append(b)
    return boxes
for year, image_set in sets:
    print(year, image_set)
    lst=[]
    with open(os.path.join('/media/amir/4C8C65E67913301E/Datasets/VOCdevkit/VOC%s/%s.txt' % (year, image_set)), 'r') as f:
        image_ids = f.read().strip().split()
    #with open(os.path.join("/media/amir/4C8C65E67913301E/Datasets/VOCdevkit", '%s_%s.txt' % (year, image_set)), 'w') as f:
    for image_id in image_ids:
        if image_set in ["train","val"]:

            path="train"
        else :
            path="test"
        current_path=('%s/VOC%s/%s/JPEGImages/%s.jpg' % ("VOCdevkit", year,path ,image_id))
        boxes=convert_annotation(year, image_id, f,image_set)
        lst.append([current_path,boxes])
    with open(os.path.join("/media/amir/4C8C65E67913301E/Datasets/VOCdevkit", '%s_%s.pkl' % (year, image_set)), 'wb') as f:
        pickle.dump(lst,f)
    print (image_set,len(lst))
        