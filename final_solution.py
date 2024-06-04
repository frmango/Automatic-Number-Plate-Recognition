from ultralytics import YOLO
from track import Sort
from scipy.interpolate import interp1d

import cv2
import csv
import string
import easyocr
import numpy as np
import pandas as pd

import argparse
import os
import sys
import ast


parser = argparse.ArgumentParser(prog="Assignment Pipeline")
parser.add_argument("--pretrained-model-path", type=str, default="./yolov8n.pt", help="Path of the pre-trained model to load before performing fine tuning.")
parser.add_argument("--train", type=str, default="no", choices=["yes", "no"], help="Whether or not to train to train a new YOLO model on specified data.")
parser.add_argument("--video-to-infer", type=str, default="./french-license-plate.mp4", help="Path of the video upon which performing inference.")
parser.add_argument("--save-period", type=int, default=1, help="Number of epochs after which training model has to be saved to disk.")
args = parser.parse_args()

# Reference: COCO dataset release in 2017
# Link: https://github.com/amikelive/coco-labels
vehicles = [2, 3, 5, 7]

# French license plates do not contain language-specific characters.
# They consist only of standard alphanumeric characters (A-Z and 0-9).
reader = easyocr.Reader(['en'], gpu = False)
tracker = Sort()

# Encountered Issue with EasyOCR:
#       Hard to tell apart some characters from digits and viceversa i.e. O and 0.  
#       The model is supposed to work regardeless potential similarities in shape between characters and digits.
char2int = {'O': '0',
            'I': '1',
            'J': '3',
            'A': '4',
            'G': '6',
            'S': '5'}

int2char = {'0': 'O',
            '1': 'I',
            '3': 'J',
            '4': 'A',
            '6': 'G',
            '5': 'S'}

def retrieve_text_score(thresh_license_plate):
    """ 
    Checks if the processed license plate is compliant to the French license plate format.
    Novelty: it takes into account potential digit-char (and vice versa) mismatches.
    If that is the case, it outputs its text -read via EasyOCR and formatted to erase potential mismatched related to the model employed- and its confidence score.

    Args:
        thresh_license_plate - Cropped and Processed i.e. greyscaled and thresholded license plate

    Output:
        License plate text and its relative confidence score
    """
    detected_texts = reader.readtext(thresh_license_plate)
    for detected_text in detected_texts:
        compliant = False
        bbox, text, conf = detected_text

        # A first formatting on text:
        #        - remove white spaces
        #        - uppercase
        text = text.upper().replace(' ', '')

        # Check if French format compliant:
        #          - seven alphanumeric characters (two letters, a dash, three numbers, a dash and two letters)
        if len(text) == 9 and text[2] == '-' and text[6] == '-':
            if  all(c in string.ascii_uppercase or c in int2char for c in text[0:2]) and all(c.isdigit() or c in char2int for c in text[3:6]) and all(c in string.ascii_uppercase or c in int2char for c in text[7:9]):
                compliant = True

            if compliant:
                # Must fix any potential mismatch wrt the French license plate format
                formatted_text = ''
                mapping = {
                                0: int2char, 1: int2char,   # two letters
                                3: char2int, 4: char2int, 5: char2int,  # three numbers
                                7: int2char, 8: int2char  # two letters
                            }
                for i, char in enumerate(text):
                    if i in mapping:
                        formatted_text += mapping[i].get(char, char)  
                    else:
                        formatted_text += char  # add dashes as they are

                return formatted_text, conf
    return None, None


# Vehicles are detected before localizing the RoI where the license plate might be located:
#                      - This reduces False Positives
#                      - Vehicle's bounding box helps in accurately predicting the license plate's
#                      - Robust technique to noises from the background
def is_within(tracked_bboxs, plate_detection):
    """
    Checks whether the detected license plate is within a previously-detected vehicle's bounding box. 
    If that is the case, vehicle-of-interest's info are returned.
    
    Args: 
        tracked_bboxs - Vehicle coordinates and its ID  
        plate_detection - License Plate tuple (x1, y1, x2, y2, score, class_id)  

    Output:
        Info about corresponding vehicle the license plate's bounding box is within
    """
    res = None

    x1, y1, x2, y2, score, class_id = plate_detection
    for vehicle in tracked_bboxs:
        # Issue: The info related to each vehicle saved in the csv file might 'jump' from one frame to another
        #        This might affect the visualization
        # Solution: Vehicle id is retrained to interpolate missing bounding boxes
        v_x1, v_y1, v_x2, v_y2, vehicle_id = vehicle 
        if (x1 > v_x1 and y1 > v_y1 and x2 < v_x2 and y2 < v_y2):
            res = vehicle
            break
    
    if res is None:
        return -1, -1, -1, -1, -1
    else:
        return res

def save_to_csv(res, output_path):
    """
    Write the results to a CSV file.

    Args:
        res (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'vehicle_id', 'vehicle_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_count in res.keys():
            for vehicle_id in res[frame_count].keys():
                print(res[frame_count][vehicle_id])
                if 'vehicle' in res[frame_count][vehicle_id].keys() and \
                   'license_plate' in res[frame_count][vehicle_id].keys() and \
                   'text' in res[frame_count][vehicle_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_count,
                                                            vehicle_id,
                                                            '[{} {} {} {}]'.format(
                                                                res[frame_count][vehicle_id]['vehicle']['bbox'][0],
                                                                res[frame_count][vehicle_id]['vehicle']['bbox'][1],
                                                                res[frame_count][vehicle_id]['vehicle']['bbox'][2],
                                                                res[frame_count][vehicle_id]['vehicle']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                res[frame_count][vehicle_id]['license_plate']['bbox'][0],
                                                                res[frame_count][vehicle_id]['license_plate']['bbox'][1],
                                                                res[frame_count][vehicle_id]['license_plate']['bbox'][2],
                                                                res[frame_count][vehicle_id]['license_plate']['bbox'][3]),
                                                            res[frame_count][vehicle_id]['license_plate']['bbox_score'],
                                                            res[frame_count][vehicle_id]['license_plate']['text'],
                                                            res[frame_count][vehicle_id]['license_plate']['text_score'])
                            )
                    
        # ISSUE 1: Frame numbers in which the vehicle and its respective license plate appear might be non-consecutive
        #          Interpolate frames in which information has not been extracted
        #                      - take the average of consecutive and corresponding coordinates in the saved csv
                    
        # ISSUE 2: There might be more than one 'text' values related to the same license plate.
        #          The one with the highest confidence score will be selected.
        f.close()

def interpolate_bounding_boxes(data):
    """
    Interpolates bounding boxes for vehicles and license plates across frames in a video sequence.

    The function extracts frame numbers, vehicle IDs, and bounding boxes for both vehicles and license plates
    from the input data. It then processes each unique vehicle ID to interpolate missing bounding boxes
    between detected frames, helping to maintain continuity of tracking despite frame skips or missed detections.
    
    Args:
        data (list of dicts): Contains the per-frame bounding box data for vehicles and their license plates.
    
    Returns:
        interpolated_data (list of dicts): The data with interpolated bounding boxes for frames missing detections.
    """
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_number']) for row in data])
    vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in data])
    vehicle_bboxes = np.array([list(map(float, row['vehicle_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_vehicle_ids = np.unique(vehicle_ids)
    for vehicle_id in unique_vehicle_ids:

        frame_numbers_ = [p['frame_number'] for p in data if int(float(p['vehicle_id'])) == int(float(vehicle_id))]
        print(frame_numbers_, vehicle_id)

        # Identifying all frames for the current vehicle
        vehicle_mask = vehicle_ids == vehicle_id
        vehicle_frame_numbers = frame_numbers[vehicle_mask]
        vehicle_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = vehicle_frame_numbers[0]
        last_frame_number = vehicle_frame_numbers[-1]

        for i in range(len(vehicle_bboxes[vehicle_mask])):
            frame_number = vehicle_frame_numbers[i]
            vehicle_bbox = vehicle_bboxes[vehicle_mask][i]
            license_plate_bbox = license_plate_bboxes[vehicle_mask][i]
            # Check if there's a gap in frames to interpolate
            if i > 0:
                prev_frame_number = vehicle_frame_numbers[i-1]
                prev_vehicle_bbox = vehicle_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    # Interpolate vehicle bounding boxes
                    interp_func = interp1d(x, np.vstack((prev_vehicle_bbox, vehicle_bbox)), axis=0, kind='linear')
                    interpolated_vehicle_bboxes = interp_func(x_new)
                    # Interpolate license plate bounding boxes
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)
                    
                    vehicle_bboxes_interpolated.extend(interpolated_vehicle_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
            # Append current frame's bounding boxes
            vehicle_bboxes_interpolated.append(vehicle_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
        
        # Create output data structure for interpolated frames
        for i in range(len(vehicle_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_number'] = str(frame_number)
            row['vehicle_id'] = str(vehicle_id)
            row['vehicle_bbox'] = ' '.join(map(str, vehicle_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))
            
            # Set additional fields based on whether the frame was interpolated
            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_number']) == frame_number and int(float(p['vehicle_id'])) == int(float(vehicle_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draws a rectangular border around a specified region of an image, with the border edges
    extending into the rectangle at specified lengths. The function draws partial borders
    at each corner, creating a "highlight" effect that does not cover the entire perimeter.

    Args:
        img - The image on which to draw the border in place
        top_left - The (x, y) coordinates of the top-left corner of the rectangle
        bottom_right - The (x, y) coordinates of the bottom-right corner of the rectangle
        color - The color of the border in BGR format (Blue, Green, Red)
        thickness - The thickness of the lines that make up the border
        line_length_x - The horizontal length of each corner line
        line_length_y - The vertical length of each corner line

    Returns:
        The image with the border drawn on it
    """
    
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def main():

    if args.train == "yes":
        # train the new model
        if not os.path.exists(args.pretrained_model_path):
            return sys.exit(-1)
        # Load a model
        model = YOLO(args.pretrained_model_path)  # build a new (nano i.e. smallest) model from scratch

        # Train the model
        model.train(
            data="./data.yaml",
            epochs=3,
            project="runs2/train",
            name="my_model",
            save_period=args.save_period
        )
    else:
        # load models for bounding box detection and license plate identification
        vehicle_detector = YOLO("./yolov8n.pt")
        plate_detector = YOLO("./best.pt")
        # load video and read frames
        cap = cv2.VideoCapture(args.video_to_infer)
        # detect and track vehicles for each frame
        frame_count = 0
        res = {}

        # 3.2 Read Frames
        while cap.isOpened():
            print(f"Frame Number: {frame_count}")
            # Read a frame from the video and check if it was successfully read
            ret, frame = cap.read()

            if not ret:
                break
            
            res[frame_count] = {}
            
            # 4.1 Detect Vehicles
            vehicles_bbox = []
            vehicle_detections = vehicle_detector(frame)[0]
            for vehicle_detection in vehicle_detections.boxes.data.tolist():
                # print(vehicle_detection)

                # x1, y1 - top left coordinates 
                # x2, y2 - bottom right coordinates
                # score  - how confident the model is about the detected object
                x1, y1, x2, y2, score, class_id = vehicle_detection
                if int(class_id) in vehicles:
                    vehicles_bbox.append([x1, y1, x2, y2, score])       # bounding boxes and scores of each vehicle in the current frame

            # 4.2 Track Vehicles
            if not vehicles_bbox:
                frame_count += 1
                continue
            tracked_bboxs = tracker.update(np.array(vehicles_bbox))     # assign an id to each distinct vehicle in the current frame
                                                                            # Rationale: same cars are sliding throughout the video
            
            # ISSUE: Tracking algorithm extending the bounding box outside the actual frame based on its motion prediction or tracking logic
            # SOLUTION: Clamping coordinates
            height, width = frame.shape[:2]
            clamped_tracked_bboxs = []
            for bbox in tracked_bboxs:
                x1, y1, x2, y2, vehicle_id = bbox
                # Clamp coordinates
                clamped_x1 = max(0, min(x1, width - 1))
                clamped_y1 = max(0, min(y1, height - 1))
                clamped_x2 = max(0, min(x2, width - 1))
                clamped_y2 = max(0, min(y2, height - 1))
                clamped_tracked_bboxs.append([clamped_x1, clamped_y1, clamped_x2, clamped_y2, vehicle_id])

            # 5. Detect License Plates
            plate_detections = plate_detector(frame)[0]
            for plate_detection in plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate_detection       # detection of every license plate in the current frame

                # 6. Assign (each) License Plate to Vehicle
                v_x1, v_y1, v_x2, v_y2, vehicle_id = is_within(clamped_tracked_bboxs, plate_detection)
                
                # 7.1 Crop License Plate
                print("#########################")
                print(f"Vehicle ID: {vehicle_id}")
                if vehicle_id != -1:  
                    license_plate = frame[int(y1) : int(y2), int(x1) : int(x2), : ]  

                    # 7.2 Process License Plate    
                    # 7.2.1 Grayscale
                    gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    # 7.2.2 Thresholding
                    #     - all pixels below 64 -> set to 255
                    #     - all pixels above 64 -> set to 0
                    _, thresh_license_plate = cv2.threshold(gray_license_plate, 64, 255, cv2.THRESH_BINARY_INV)

                    # 8. Read License Plate Number
                    text, conf = retrieve_text_score(thresh_license_plate)

                    # 9. Save Results
                    if text is not None:
                        res[frame_count][vehicle_id] = {'vehicle': {'bbox': [v_x1, v_y1, v_x2, v_y2]},
                                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': text,
                                                                            'bbox_score': score,
                                                                            'text_score': conf}
                                                        }

            frame_count += 1
                
        cap.release()
        # save results
        save_to_csv(res, "./result.csv")
        # Load the CSV file
        with open('result.csv', 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        # Interpolate missing data
        interpolated_data = interpolate_bounding_boxes(data)

        # Write updated data to a new CSV file
        header = ['frame_number', 'vehicle_id', 'vehicle_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open('result_interpolated.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)

        results = pd.read_csv('./result_interpolated.csv')

        # Load video
        video_path = args.video_to_infer
        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sol = cv2.VideoWriter('./sol.mp4', fourcc, fps, (width, height))

        license_plate = {}
        for vehicle_id in np.unique(results['vehicle_id']):
            max_ = np.amax(results[results['vehicle_id'] == vehicle_id]['license_number_score'])
            license_plate[vehicle_id] = {'license_crop': None,
                                    'license_plate_number': results[(results['vehicle_id'] == vehicle_id) &
                                                                    (results['license_number_score'] == max_)]['license_number'].iloc[0]}
            cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['vehicle_id'] == vehicle_id) &
                                                    (results['license_number_score'] == max_)]['frame_number'].iloc[0])
            ret, frame = cap.read()

            x1, y1, x2, y2 = ast.literal_eval(results[(results['vehicle_id'] == vehicle_id) &
                                                    (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

            license_plate[vehicle_id]['license_crop'] = license_crop


        frame_number = -1

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read frames
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_number += 1
            if ret:
                df_ = results[results['frame_number'] == frame_number]
                for row_indx in range(len(df_)):
                    # Draw vehicle
                    v_x1, v_y1, v_x2, v_y2 = ast.literal_eval(df_.iloc[row_indx]['vehicle_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    draw_border(frame, (int(v_x1), int(v_y1)), (int(v_x2), int(v_y2)), (0, 255, 0), 25,
                                line_length_x=200, line_length_y=200)

                    # Draw license plate
                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                    # Crop license plate
                    license_crop = license_plate[df_.iloc[row_indx]['vehicle_id']]['license_crop']

                    H, W, _ = license_crop.shape

                    try:
                        frame[int(v_y1) - H - 100:int(v_y1) - 100,
                            int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = license_crop

                        frame[int(v_y1) - H - 400:int(v_y1) - H - 100,
                            int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            17)

                        cv2.putText(frame,
                                    license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_number'],
                                    (int((v_x2 + v_x1 - text_width) / 2), int(v_y1 - H - 250 + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    (0, 0, 0),
                                    17)

                    except:
                        pass

                sol.write(frame)
                frame = cv2.resize(frame, (1280, 720))

        sol.release()
        cap.release()

if __name__ == "__main__":
    main()