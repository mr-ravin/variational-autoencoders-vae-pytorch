import cv2
import numpy as np
import random
import os

def run():
    for count in range(100):
        template = np.ones((128,128,3))*255
        template = template.astype(np.uint8) 
        radius = int(random.uniform(0.2, 0.9)*64)
        cv2.circle(template, (64,64), radius=radius, color=(0,0,0), thickness=-1)
        cv2.imwrite("dataset/train/circle_"+str(count)+".jpg", template)

    for count in range(100, 130):
        template = np.ones((128,128,3))*255
        template = template.astype(np.uint8) 
        radius = int(random.uniform(0.2, 0.9)*64)
        cv2.circle(template, (64,64), radius=radius, color=(0,0,0), thickness=-1)
        cv2.imwrite("dataset/valid/circle_"+str(count)+".jpg", template)

    for count in range(130,140):
        template = np.ones((128,128,3))*255
        template = template.astype(np.uint8) 
        radius = int(random.uniform(0.2, 0.9)*64)
        cv2.circle(template, (64,64), radius=radius, color=(0,0,0), thickness=-1)
        cv2.imwrite("dataset/test/circle_"+str(count)+".jpg", template)