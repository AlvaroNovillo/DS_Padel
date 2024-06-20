# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:30:44 2024

@author: pc
"""


from ultralytics import YOLO 
import cv2
import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import measure_distance, get_center_of_bbox,get_intersection_1

# Check if TensorFlow is using the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs found. TensorFlow is using the CPU.")


class PlayerTracker:
    def __init__(self, box_model_path = 'yolov8x',pose_model_path_or_url = 'https://bit.ly/metrabs_l', skeleton = 'smpl+head_30'):

        self.pose_model = tfhub.load(pose_model_path_or_url)
        self.box_model = YOLO(box_model_path).to('cuda') 
        self.skeleton = skeleton
        self.joint_names = self.pose_model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        self.joint_edges = self.pose_model.per_skeleton_joint_edges[skeleton].numpy()
        self.frame_count = -1

    def choose_and_filter_players(self, court_keypoints, player_detections,n):
        player_detections_first_frame = player_detections[0]
        player_detections_first_frame = player_detections_first_frame[0] #First frame
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame,n)
        filtered_player_detections = []
        
        self.frame_count = -1
        for player_dict in player_detections:
            self.frame_count += 1
            filtered_frame_dict = {}
            filtered_frame_dict[self.frame_count] = {}
            frame_dict = player_dict[self.frame_count]
            filtered_frame_dict[self.frame_count] = {key: value for key, value in frame_dict.items() if key in chosen_player}
            filtered_player_detections.append(filtered_frame_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_detections_first_frame,n):
        distances = []
        for track_id, values in player_detections_first_frame.items():
            bbox = values.get('boxes',None)
            bbox = bbox.numpy()
            bbox[2], bbox[3] = bbox[2] + bbox[0], bbox[3] + bbox[1]
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')

            #Court keypoints

            #x1, y1, x2, y2 = court_bbox
            court_center = get_intersection_1(court_keypoints[0],court_keypoints[2],court_keypoints[1],court_keypoints[3])
            distance = measure_distance(player_center, court_center)
            if distance < min_distance:
                min_distance = distance
            distances.append((track_id, min_distance))

        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first n tracks
        chosen_players = [distance[0] for distance in distances[:n]]
        return chosen_players
       
    def detect_frames(self,frames, n, court_detections, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        player_detections = self.choose_and_filter_players(court_detections,player_detections,n)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):

        #Detect players bounding boxes
        results = self.box_model.track(frame, persist=True)[0]
        id_name_dict = results.names

        #initialize dataframes
        boxes = []
        player_dict = {}
        self.frame_count +=1 
        print(self.frame_count) 
        player_dict[self.frame_count] = {}
        id_list = []

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                # Append bounding box in [left, top, width, height] format
                boxes.append([x1, y1, w, h])
                id_list.append(track_id)

        boxes = tf.constant(np.array(boxes), dtype=tf.float32)
        #Extract player poses
        results = self.pose_model.estimate_poses(frame, boxes, skeleton=self.skeleton)

        
        for box, poses2d, poses3d, track_id in zip(boxes, results['poses2d'], results['poses3d'], id_list):
            player_dict[self.frame_count][track_id] = {'boxes': box,
                                      'poses2d': poses2d,
                                      'poses3d': poses3d}
            
        
        return player_dict
    

    def draw_bboxes(self,video_frames, player_detections,do_3d = False,do_pe = False):
        output_video_frames = []
        self.frame_count = -1
        for frame, player_dict in zip(video_frames, player_detections):
            self.frame_count += 1
            frame_dict = player_dict[self.frame_count]
            if do_3d:
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 2, 1)
                pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
                for track_id, values in frame_dict.items():
                    image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    image_ax.imshow(image)
                    
                    #Values
                    bbox = values.get('boxes',None)
                    pose2d = values.get('poses2d',None)
                    pose3d = values.get('poses3d', None)

                    # Draw Bounding Boxes
                    x, y, w, h = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
                    player_id_text = f"Player ID: {track_id}"
                    image_ax.text(int(bbox[0]), int(bbox[1]) - 10, player_id_text, fontsize=9, color='k', fontweight='bold')

                    #Draw pose detections
                    pose_ax.view_init(5, -75)
                    #pose_ax.set_xlim3d(-2500, 2500)
                    #pose_ax.set_zlim3d(-2500, 2500)
                    #pose_ax.set_ylim3d(2000, 5000)
                    if not None in (bbox, pose2d, pose3d):
                        pose3d = pose3d.numpy()
                        pose2d = pose2d.numpy()

                        pose3d[..., 1], pose3d[..., 2] = pose3d[..., 2], -pose3d[..., 1]
                        for i_start, i_end in self.joint_edges:
                            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
                            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
                        image_ax.scatter(*pose2d.T, s=2)
                        pose_ax.scatter(*pose3d.T, s=2)

                
                fig.canvas.draw()

                img_plot = np.array(fig.canvas.renderer.buffer_rgba())

                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                                            
                output_video_frames.append(mat_frame)

                plt.close()
            else:
                fig = plt.figure(figsize=(10, 5.2))
                image_ax = fig.add_subplot(1, 1, 1)
                for track_id, values in frame_dict.items():
                    image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    image_ax.imshow(image)
                    
                    #Values
                    bbox = values.get('boxes',None)
                    

                    # Draw Bounding Boxes
                    x, y, w, h = bbox
                    image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
                    player_id_text = f"Player ID: {track_id}"
                    image_ax.text(int(bbox[0]), int(bbox[1]) - 10, player_id_text, fontsize=9, color='k', fontweight='bold')
                    if do_pe:

                        pose2d = values.get('poses2d',None)
                        if not None in (bbox, pose2d):
                            pose2d = pose2d.numpy()
                            for i_start, i_end in self.joint_edges:
                                image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
                            image_ax.scatter(*pose2d.T, s=2) 

                fig.canvas.draw()

                img_plot = np.array(fig.canvas.renderer.buffer_rgba())

                mat_frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                                            
                output_video_frames.append(mat_frame)

                plt.close()

        
        return output_video_frames