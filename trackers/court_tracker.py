import cv2
import pickle

from utils import import_court,court_borders,get_intersection_1,order_points,homography_scorer
import random
import numpy as np
import matplotlib.pyplot as plt



class CourtTracker:
    def __init__(self):
        self.court_reference = import_court()
        self.average_mask = None

    def image_preprocessing(self,frame,k = 2, plot = True):
        # Image preprocessing with Kmeans (K = 2)
        Z = frame.reshape((-1, 3))
        Z = np.float32(Z)

        # Define criteria, number of clusters(K) and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = k
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back into uint8, and reshape to the original image
        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((frame.shape))

        # Get the label for the pixel in the middle (court label)
        middle_pixel_coords = (frame.shape[0] // 2, frame.shape[1] // 2)
        middle_pixel_index = middle_pixel_coords[0] * frame.shape[1] + middle_pixel_coords[1]
        court_label = label[middle_pixel_index]

        # Create a mask for the court
        mask = np.abs((label == court_label).astype(np.uint8))
        mask = mask.reshape((frame.shape[0], frame.shape[1]))
        #cv2.imwrite('mask_no_pre.png',mask*255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = cv2.blur(mask, (30, 30))

        if plot:
            cv2.imshow("Mask", mask * 255)
            cv2.waitKey(0)

        return mask
    
    def compute_average_mask(self, frames, N=5):
        selected_indices = random.sample(range(len(frames)), N)
        mask_sum = None
        for i in selected_indices:
            frame = frames[i]
            mask = self.image_preprocessing(frame, plot=True)
            if mask_sum is None:
                mask_sum = np.zeros_like(mask, dtype=np.float32)
            mask_sum += mask
        average_mask = (mask_sum / N).astype(np.uint8)
        self.average_mask = average_mask
        #cv2.imwrite('mask_blur.png',average_mask*255)
        
    
    def contour_detection(self,mask,frame,plot = False):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt = contours[0]
        if plot:    
            frame = cv2.drawContours(frame.copy(), cnt, -1, (0, 255, 0), 5)
            cv2.imshow('contours', frame)
            cv2.waitKey(0)

        # Extract coordinates of the contour
        contour_points = cnt[:, 0, :]   

        return contour_points


    def process_court_contour(self,contour_points, frame, ref_borders, court_reference, plot=False):
        # Find the extreme points to approximate the four sides
        top_points = contour_points[contour_points[:, 1] < frame.shape[0] // 2.7]
        bottom_points = contour_points[contour_points[:, 1] > frame.shape[0] // 1.5]

        # Find the extreme points in x from the top and bottom points
        extreme_left_point = bottom_points[np.argmin(bottom_points[:, 0])]
        extreme_right_point = bottom_points[np.argmax(bottom_points[:, 0])]
        extreme_left_point_1 = top_points[np.argmin(top_points[:, 0])]
        extreme_right_point_1 = top_points[np.argmax(top_points[:, 0])]

        delta_x = 200  # Number of pixels to extend

        # Function to calculate slope
        def calculate_slope(p1, p2):
            return (p2[1] - p1[1]) / (p2[0] - p1[0])

        # Function to extend a line
        def extend_line(p1, p2, delta_x):
            slope = calculate_slope(p1, p2)
            extended_p1 = (p1[0] - delta_x, int(p1[1] - slope * delta_x))
            extended_p2 = (p2[0] + delta_x, int(p2[1] + slope * delta_x))
            return extended_p1, extended_p2

        # Draw the lines on the result image
        extended_right_p1, extended_right_p2 = extend_line(extreme_right_point, extreme_right_point_1, -delta_x)
        #frame = cv2.line(frame, extended_right_p1, extended_right_p2, (0, 0, 255), 2)

        extended_left_p1, extended_left_p2 = extend_line(extreme_left_point, extreme_left_point_1, delta_x)
        #frame = cv2.line(frame, extended_left_p1, extended_left_p2, (0, 0, 255), 2)

        #frame = cv2.line(frame, (0, int(np.min(top_points[:, 1]))), (frame.shape[1], int(np.min(top_points[:, 1]))), (0, 0, 255), 2)
        #frame = cv2.line(frame, (0, int(np.max(bottom_points[:, 1]))), (frame.shape[1], int(np.max(bottom_points[:, 1]))), (0, 0, 255), 2)

        i1 = get_intersection_1(extended_right_p1, extended_right_p2, (0, int(np.min(top_points[:, 1]))), (frame.shape[1], int(np.min(top_points[:, 1]))))
        i2 = get_intersection_1(extended_right_p1, extended_right_p2, (0, int(np.max(bottom_points[:, 1]))), (frame.shape[1], int(np.max(bottom_points[:, 1]))))
        i3 = get_intersection_1(extended_left_p1, extended_left_p2, (0, int(np.min(top_points[:, 1]))), (frame.shape[1], int(np.min(top_points[:, 1]))))
        i4 = get_intersection_1(extended_left_p1, extended_left_p2, (0, int(np.max(bottom_points[:, 1]))), (frame.shape[1], int(np.max(bottom_points[:, 1]))))

        intersections = np.array([i1, i2, i3, i4])
        intersections = order_points(intersections)

        homography_matrix = cv2.getPerspectiveTransform(ref_borders, intersections)
        court = homography_scorer(homography_matrix, court_reference, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if plot:
            plot_court = np.where(court == 1)

            fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
            ax = axes.ravel()

            for i in intersections:
                ax[0].scatter(i[0], i[1], color='r', s=30)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            ax[0].imshow(frame)
            #ax[0].set_title('Candid Court Borders', fontsize=26)

            ax[1].scatter(plot_court[1], plot_court[0], color='g', s=1)
            ax[1].set_xlim(0, frame.shape[1])
            ax[1].set_ylim(0, frame.shape[0])
            ax[1].invert_yaxis()
            ax[1].imshow(frame)
            #ax[1].set_title('Applied Homography', fontsize=26)




            plt.tight_layout()
            plt.show()

        return homography_matrix, intersections


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        frame = frames[0]
        ref_borders = court_borders(self.court_reference,court_factor = 1)

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                court_detections = pickle.load(f)
            return court_detections

        #Image preprocessing (create average mask)
        self.compute_average_mask(frames)
        homography_matrix, court_detections = self.detect_frame(frame, self.average_mask,self.court_reference, ref_borders = ref_borders)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(court_detections, f)
        
        return homography_matrix, court_detections

    def detect_frame(self,frame, mask,court_reference, ref_borders):
        

        # Court detection using Contour detection

        court_outline = self.contour_detection(mask,frame,plot = False)

        #Border detection and 4-point homography transformation

        homography_matrix, court_detections = self.process_court_contour(court_outline, frame, ref_borders, court_reference, plot=False)


        return homography_matrix, court_detections



    def draw_bboxes(self, video_frames, court_detections,homography_matrix):
        output_video_frames = []
        k = 0
        row, column = np.where(court_detections == 1)
        for frame in video_frames:
            # Draw frame number on top left corner
            k += 1
            cv2.putText(frame, f"Frame: {k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in court_detections:
                cv2.circle(frame, tuple(i.astype(int)), 20, (0, 0, 255), -1)
            output_video_frames.append(frame)
        
        return output_video_frames

