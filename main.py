# -*- coding: utf-8 -*-
"""
Padel analyzer
        TODO:

            - APPLY PLAYER FILTER 
            - PARALELIZE PLAYER, BALL AND COURT MODEL
            - IMPLEMENT COURT MODEL
            - 
@author: pc
"""
from utils import (read_video, 
                   save_video)

import cv2

from trackers import (PlayerTracker,
                      BallTracker,
                      CourtTracker)

from minicourt import Court2D


def main():
    #Read video
    input_video = "main.mp4"
    video_frames = read_video(input_video)


    #Load the models 
    court_tracker = CourtTracker()
    
    player_tracker = PlayerTracker()

    ball_tracker = BallTracker(model_path = 'models/model_best.pt')


    print("--- Tracking players, poses and the ball ---")

    

    #Detect pitch
    court_detected = False
    #Since the frame sampling is now being made randomly, we can try to repeat the homography process in case it fails
    while not court_detected:
        try:
            homography_matrix, court_detections = court_tracker.detect_frames(video_frames, 
                                                            read_from_stub = False,
                                                            stub_path="tracker_stubs/court_detections.pkl"
                                                            )
            court_detected = True
        except:
            print("--- Court not detected with the current sampling. Trying again... ---")
            homography_matrix, court_detections = court_tracker.detect_frames(video_frames, 
                                                            read_from_stub = False,
                                                            stub_path="tracker_stubs/court_detections.pkl"
                                                            )
    #Detect players
    
    player_detections = player_tracker.detect_frames(frames = video_frames, 
                                                     n = 4,
                                                     court_detections = court_detections,
                                                     read_from_stub = False,
                                                     stub_path="tracker_stubs/player_detections_main.pkl"
                                                     )
    
    
    
    #Detect ball
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                     read_from_stub= False,
                                                     stub_path="tracker_stubs/ball_detections_main.pkl",
                                                     homography_matrix = homography_matrix

                                                     )
    
    
    #Output video
    

    print("--- Preparing 3d output video ---")

    #Output video
   
    #Court detections
    court_video_frames = court_tracker.draw_bboxes(video_frames, court_detections,homography_matrix)

    #Ball detections
    ball_video_frames = ball_tracker.draw_bboxes(court_video_frames, ball_detections)
    
    #Player detections
    output_video_frames = player_tracker.draw_bboxes(ball_video_frames, player_detections,do_3d = False, do_pe = False)

      
    #Save the final video
    save_video(output_video_frames, "output_videos/result_main.avi",fps = cv2.VideoCapture(input_video).get(cv2.CAP_PROP_FPS))


    # 2D court representation

    print("--- Preparing 2d tracking video ---")

    court2d = Court2D(cv2.VideoCapture(input_video))
    
    court2d.plot_positions(player_detections,ball_detections,homography_matrix,output_csv_path = 'data/tracking.csv',output_video_path ="output_videos/result_2d_main.avi")
   
  
    
if __name__ == '__main__':
    
    main()