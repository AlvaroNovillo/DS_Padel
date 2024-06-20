import cv2
import numpy as np
import csv
import pandas as pd

from utils import (
    import_court,
    get_foot_position,

)
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

class Court2D:
    def __init__(self, input_video):
        self.fps = input_video.get(cv2.CAP_PROP_FPS)
        self.frame_count = -1


    def interpolate_ball_positions(self, ball_detections):
        x_ball, y_ball = [d[0] for d in ball_detections], [d[1] for d in ball_detections]
        is_none = [int(x is None) for x in x_ball]
        interp = 3
        max_dist = 10  # Define the maximum allowable distance to consider a point as an outlier
        counter = 0
        
        for num in range(interp, len(x_ball) - 1):
            if x_ball[num] is None and sum(is_none[num - interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num - interp:num], y_ball[num - interp:num])
                if x_ext is not None and y_ext is not None:
                    x_ball[num] = x_ext
                    y_ball[num] = y_ext
                    is_none[num] = 0
                    if x_ball[num + 1] is not None:
                        dist = distance.euclidean((x_ext, y_ext), (x_ball[num + 1], y_ball[num + 1]))
                        if dist > max_dist:
                            x_ball[num + 1], y_ball[num + 1], is_none[num + 1] = None, None, 1
                    counter += 1
                else:
                    is_none[num] = 1
            else:
                counter = 0  

        ball_detections = [[x, y] for x, y in zip(x_ball, y_ball)]
        return ball_detections

    def extrapolate(self, x_coords, y_coords):
        valid_indices = [i for i, x in enumerate(x_coords) if x is not None and y_coords[i] is not None]
        valid_xs = [x_coords[i] for i in valid_indices]
        valid_ys = [y_coords[i] for i in valid_indices]
        
        if len(valid_xs) < 2:
            return None, None

        func_x = CubicSpline(valid_indices, valid_xs, bc_type='natural')
        func_y = CubicSpline(valid_indices, valid_ys, bc_type='natural')
        x_ext = func_x(len(x_coords))
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)

    
    def apply_homography(self, position, homography_matrix):
        homography_matrix = np.linalg.inv(homography_matrix)
        position_h = np.array([position[0], position[1], 1]).flatten()
        transformed_position_h = np.dot(homography_matrix, position_h)
        transformed_position = transformed_position_h[:2] / transformed_position_h[2]
        return int(transformed_position[0]), int(transformed_position[1])

    def convert_to_meters(self, position, pixel_to_meter_ratio):
        return position[0] * pixel_to_meter_ratio[0], position[1] * pixel_to_meter_ratio[1]

    def plot_positions(self, player_detections, ball_detections, homography_matrix, output_csv_path,output_video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, self.fps, (1080, 1920))
        
        pixel_to_meter_ratio = (20.1168 / 1080, 10.0584 / 1920)  # Width and Height ratios in meters per pixel
        #trace_length = 7  # Length of the trace

        #ball_detections = self.interpolate_ball_positions(ball_detections)

        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "PlayerID", "X_meters", "Y_meters", "BallX_meters", "BallY_meters"])

            for frame_index, player_dict in enumerate(player_detections):
                self.frame_count += 1
                court_reference = import_court()
                frame_dict = player_dict[self.frame_count]
                for player_id, values in frame_dict.items():
                    bbox = values.get('boxes', None)
                    if bbox is not None:
                        # bounding box in [left, top, right, bottom] format
                        bbox = bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]
                        foot_position = get_foot_position(bbox)
                        transformed_position = self.apply_homography(foot_position, homography_matrix)
                        x_meters, y_meters = self.convert_to_meters(transformed_position, pixel_to_meter_ratio)
                        
                        cv2.circle(court_reference, transformed_position, 10, (0, 0, 255), -1)
                        cv2.putText(court_reference, f"{player_id}", (int(transformed_position[0]), int(transformed_position[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Write player data to CSV
                        writer.writerow([self.frame_count, player_id, x_meters, y_meters, "", ""])

                """
                for trace_index in range(trace_length):
                    if frame_index - trace_index >= 0:
                        ball_pos = ball_detections[frame_index - trace_index]
                        if ball_pos[0] is not None:
                            x = int(ball_pos[0])
                            y = int(ball_pos[1])
                            transformed_position = self.apply_homography((x, y), homography_matrix)
                            x_meters, y_meters = self.convert_to_meters(transformed_position, pixel_to_meter_ratio)
                            color_intensity = 255 - int(255 * (trace_index / trace_length))
                            cv2.circle(court_reference, (x, y), 10, (0, color_intensity, 0), -1)

                            if trace_index == 0:
                                # Write ball data to CSV for the current frame
                                writer.writerow([self.frame_count, "", "", "", x_meters, y_meters])
                        else:
                            break
                    else:
                        writer.writerow([self.frame_count, "", "", "", np.nan, np.nan])

                """


                court_reference = cv2.rotate(court_reference, cv2.ROTATE_90_CLOCKWISE)
                court_reference = cv2.flip(court_reference, 1)
                video.write(cv2.resize(court_reference, (1080, 1920)))

        cv2.destroyAllWindows()
        video.release()
