import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

def train_video_network(video_path, model_path, output_path, start_frame=0, end_frame=None, cuda=True)
    print('start training')

    # read video
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0] +'.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    face_detector = dlib.get_frontal_face_detector()

    model = model_selection(modelname='resnet18', num_out_classes=2)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()


    pbar = tqdm(total=num_frames)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1
        pbar.update(1)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------
