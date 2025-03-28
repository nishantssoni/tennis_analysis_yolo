import cv2
import sys
import numpy as np
sys.path.append("../")
import constants
from utils import (
    convert_meters_to_pixel, 
    convert_pixel_to_meters, 
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    get_center_of_bbox,
    mesure_distance,
    measure_xy_distance
)

class MiniCourt:

    def __init__(self, frame):
        self.drawing_rect_width = 250
        self.drawing_rect_hight = 500
        self.buffer = 50
        self.padding = 20
        self.set_canvas_bg_box(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()
        
    def m2p(self, meters):
        """it convet meters to pixels for minicourt only"""
        return convert_meters_to_pixel(meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)

    def p2m(self, pixels):
        """it convet pixels to meters for minicourt only"""
        return convert_pixel_to_meters(pixels,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_court_drawing_keypoints(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.m2p(constants.HALF_COURT_LINE_HEIGHT*2)
        
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.m2p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.m2p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.m2p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.m2p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.m2p(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.m2p(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.m2p(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.m2p(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points
    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y -self.padding
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_bg_box(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rect_hight
        self.start_x = self.end_x - self.drawing_rect_width
        self.start_y = self.end_y - self.drawing_rect_hight

    def draw_court(self, frame):
        # draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        # draw court line
        for i in self.lines:
            pt1 = (int(self.drawing_key_points[i[0]*2]), int(self.drawing_key_points[i[0]*2+1]))
            pt2 = (int(self.drawing_key_points[i[1]*2]), int(self.drawing_key_points[i[1]*2+1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # # draw center line
        pt0 = (int(self.drawing_key_points[0*2]), int(self.drawing_key_points[0*2+1]))
        pt1 = (int(self.drawing_key_points[1*2]), int(self.drawing_key_points[1*2+1]))
        pt2 = (int(self.drawing_key_points[2*2]), int(self.drawing_key_points[2*2+1]))
        pt3 = (int(self.drawing_key_points[3*2]), int(self.drawing_key_points[3*2+1]))
        # find middle of two points
        middle_pt1 = (pt0[0], (pt0[1] + pt2[1]) // 2)
        middle_pt2 = (pt3[0], (pt1[1] + pt3[1]) // 2)
        cv2.line(frame, middle_pt1, middle_pt2, (0, 0, 255), 2)

        return frame

    def draw_background_rect(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out
    
    def draw_mini_court(self, frames):
        output_video_frames = []
        for frame in frames:
            frame = self.draw_background_rect(frame)
            frame = self.draw_court(frame)
            output_video_frames.append(frame)
        return output_video_frames
    
    # def get_mini_court_cordinates(self,
    #                               objects_position,
    #                               closest_keypoint,
    #                               closest_keypoint_index,
    #                               player_height_in_pixel,
    #                               player_height_in_meter):
    #     # get the distance
    #     distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(objects_position,closest_keypoint)
    #     # convert pixels to meters
    #     distance_from_keypoint_x_meters = convert_pixel_to_meters(distance_from_keypoint_x_pixels, player_height_in_pixel, player_height_in_meter)
    #     distance_from_keypoint_y_meters = convert_pixel_to_meters(distance_from_keypoint_y_pixels, player_height_in_pixel, player_height_in_meter)

    #     # converts to minicourt cordinates
    #     mini_court_x_distance_pixels = self.m2p(distance_from_keypoint_x_meters)
    #     mini_court_y_distance_pixels = self.m2p(distance_from_keypoint_y_meters)
    #     closest_mini_courts_keypoint = (self.drawing_key_points[closest_keypoint_index*2],
    #                                     self.drawing_key_points[closest_keypoint_index*2+1])
        
    #     mini_court_player_position = (closest_mini_courts_keypoint[0] + mini_court_x_distance_pixels,
    #                                   closest_mini_courts_keypoint[1] + mini_court_y_distance_pixels)
    #     return mini_court_player_position

    def get_mini_court_cordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.m2p(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.m2p(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def convert_bbox_to_mini_court_cordinates(self, player_bboxs, ball_bboxs, orginal_court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT,
            2: constants.PLAYER_2_HEIGHT
        }

        output_player_bboxs = []
        output_ball_bboxs = []

        for frame_no, player_bbox in enumerate(player_bboxs):
            ball_box = ball_bboxs[frame_no][1]
            # print("\n\n\n\n", player_bbox.keys(),"\n\n\n\n")
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: mesure_distance(ball_position, get_center_of_bbox(player_bbox[x])))
            # print("\n\n\n\n", ball_position,"\n\n\n\n")
            output_player_bobox_dict = {}
            

            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # get the closest keypoints in pixels
                closest_keypoints_index = get_closest_keypoint_index(foot_position, orginal_court_keypoints,[0,2,12,13])
                closest_keypoint = (orginal_court_keypoints[closest_keypoints_index*2], 
                                    orginal_court_keypoints[closest_keypoints_index*2+1])

                # get player height in pixels
                frame_index_min = max(0, frame_no - 20)
                frame_index_max = min(len(player_bboxs), frame_no + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_bboxs[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_cordinates(foot_position,
                                                                            closest_keypoint,
                                                                            closest_keypoints_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id])
                
        

                output_player_bobox_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, orginal_court_keypoints,[0,2,12,13])
                    closest_keypoint = (orginal_court_keypoints[closest_keypoint_index*2],
                                        orginal_court_keypoints[closest_keypoint_index*2+1])
                    mini_court_ball_position = self.get_mini_court_cordinates(ball_position,
                                                                            closest_keypoint,
                                                                            closest_keypoint_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id])
                    
                    output_ball_bboxs.append({1:mini_court_ball_position})
            output_player_bboxs.append(output_player_bobox_dict)
        return output_player_bboxs, output_ball_bboxs
    

    # def convert_bbox_to_mini_court_cordinates(self,player_boxes, ball_boxes, original_court_key_points ):
    #     player_heights = {
    #         1: constants.PLAYER_1_HEIGHT,
    #         2: constants.PLAYER_2_HEIGHT
    #     }

    #     output_player_boxes= []
    #     output_ball_boxes= []

    #     for frame_num, player_bbox in enumerate(player_boxes):
    #         ball_box = ball_boxes[frame_num][1]
    #         ball_position = get_center_of_bbox(ball_box)
    #         closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: mesure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

    #         output_player_bboxes_dict = {}
    #         for player_id, bbox in player_bbox.items():
    #             foot_position = get_foot_position(bbox)

    #             # Get The closest keypoint in pixels
    #             closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
    #             closest_key_point = (original_court_key_points[closest_key_point_index*2], 
    #                                  original_court_key_points[closest_key_point_index*2+1])

    #             # Get Player height in pixels
    #             frame_index_min = max(0, frame_num-20)
    #             frame_index_max = min(len(player_boxes), frame_num+50)
    #             bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
    #             max_player_height_in_pixels = max(bboxes_heights_in_pixels)

    #             mini_court_player_position = self.get_mini_court_cordinates(foot_position,
    #                                                                         closest_key_point, 
    #                                                                         closest_key_point_index, 
    #                                                                         max_player_height_in_pixels,
    #                                                                         player_heights[player_id]
    #                                                                         )
                
    #             output_player_bboxes_dict[player_id] = mini_court_player_position

    #             if closest_player_id_to_ball == player_id:
    #                 # Get The closest keypoint in pixels
    #                 closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
    #                 closest_key_point = (original_court_key_points[closest_key_point_index*2], 
    #                                     original_court_key_points[closest_key_point_index*2+1])
                    
    #                 mini_court_player_position = self.get_mini_court_cordinates(ball_position,
    #                                                                         closest_key_point, 
    #                                                                         closest_key_point_index, 
    #                                                                         max_player_height_in_pixels,
    #                                                                         player_heights[player_id]
    #                                                                         )
    #                 output_ball_boxes.append({1:mini_court_player_position})
    #         output_player_boxes.append(output_player_bboxes_dict)

    #     return output_player_boxes , output_ball_boxes
    
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames


    # def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
    #     frame_count = min(len(frames), len(positions))
    #     for frame_num in range(frame_count):
    #         if frame_num < len(positions):
    #             for _, position in positions[frame_num].items():
    #                 x, y = position
    #                 x = int(x)
    #                 y = int(y)
    #                 cv2.circle(frames[frame_num], (x, y), 5, color, -1)
    #     return frames
