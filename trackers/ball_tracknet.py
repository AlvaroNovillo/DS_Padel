from models import BallTrackerNet
import torch
import cv2
from utils import postprocess
from tqdm import tqdm
import numpy as np
from itertools import groupby
from scipy.spatial import distance
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.model = self.model.to('cuda')
        self.model.eval()
        frame_idx = -1

    def remove_outliers(self, ball_detections, dists, max_dist=100):
        """ Remove outliers from model prediction
        :params
            ball_detections: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
            max_dist: maximum distance between two neighbouring ball points
        :return
            ball_detections: list of ball points
        """
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_detections[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_detections[i - 1] = (None, None)
        return ball_detections

    def split_track(self, ball_detections, max_gap=4, max_dist_gap=80, min_track=5):
        """ Split ball track into several subtracks in each of which we will perform
        ball interpolation.
        :params
            ball_detections: list of detected ball points
            max_gap: maximun number of coherent None values for interpolation
            max_dist_gap: maximum distance at which neighboring points remain in one subtrack
            min_track: minimum number of frames in each subtrack
        :return
            result: list of subtrack indexes
        """
        list_det = [0 if x[0] else 1 for x in ball_detections]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_detections[cursor - 1], ball_detections[cursor + l])
                if (l >= max_gap) | (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result

    def interpolation(self, coords):
        """ Run ball interpolation in one subtrack
        :params
            coords: list of ball coordinates of one subtrack
        :return
            track: list of interpolated ball coordinates of one subtrack
        """
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        track = list(zip(x, y))
        return track

    def detect_frames(self, frames,homography_matrix, read_from_stub=False, stub_path=None):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
            model: pretrained model
        :return
            ball_detections: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
        """

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        original_height, original_width = frames[0].shape[:2]
        height = 360
        width = 640
        dists = [-1] * 2
        scale = original_width / width
        ball_detections = [(None, None)] * 2
        for frame_idx in tqdm(range(2, len(frames))):
            

            img = cv2.resize(frames[frame_idx], (width, height))
            img_prev = cv2.resize(frames[frame_idx - 1], (width, height))
            img_preprev = cv2.resize(frames[frame_idx - 2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to('cuda'))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(output,scale=scale)
            ball_detections.append((x_pred, y_pred))

            if ball_detections[-1][0] is not None and ball_detections[-2][0] is not None:
                dist = distance.euclidean(ball_detections[-1], ball_detections[-2])
            else:
                dist = -1
            dists.append(dist)

        ball_detections = self.remove_outliers(ball_detections, dists)

        subtracks = self.split_track(ball_detections)
        for r in subtracks:
            ball_subtrack = ball_detections[r[0]:r[1]]
            ball_subtrack = self.interpolation(ball_subtrack)
            ball_detections[r[0]:r[1]] = ball_subtrack

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def draw_bboxes(self, video_frames, ball_track, trace=7):
        output_video_frames = []
        for frame_idx,frame in enumerate(video_frames):
            for i in range(trace):
                if frame_idx - i > 0:
                    if ball_track[frame_idx - i][0] is not None:
                        x = int(ball_track[frame_idx - i][0])
                        y = int(ball_track[frame_idx - i][1])
                        cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
                    else:
                        break
            output_video_frames.append(frame)

        return output_video_frames

