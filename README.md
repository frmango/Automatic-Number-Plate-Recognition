# Automatic Number Plate Recognition

## Project Overview

This project is tasked with implementing a comprehensive pipeline that
processes a provided video, applies vehicle detection and license plate
recognition. The system will utilize computer vision and deep learning
techniques to ensure the precise identification and recognition of
vehicles and their respective license plates. This is aimed at
showcasing each detected vehicle with a unique identifier, the
recognized license plate number and its relative detection confidence
coefficient.

## Models Employed

### Vehicle Detection Model

The vehicle detection component of this project employs YOLOv8, a
state-of-the-art object detection model, trained on the COCO dataset as
released in 2017. This model is configured to recognize only
vehicle-related categories, specifically car, motorcycle, bus and truck.
All other categories are excluded to focus on the relevant targets that
transit on streets. This strategic choice ensures that the system
optimally identifies and tracks vehicles in diverse urban environments.

### License Plate Detection Model

For license plate detection, a pre-trained YOLOv8 nano model has been
further refined to better recognize license plates. The model was
retrained on a specifically curated dataset from Roboflow,
consisting of 21,174 training images, 2,048 validation images, and 1,020
test images. The images in this dataset depict vehicles along with their
license plates, ensuring that the model can accurately identify and
process plate information under various conditions.  
The rationale that drove this dataset choice is that the dataset
underwent some pre-processing and data augmentation, more specifically:

-   Pre-processing
    -   Auto-Orient: ensures that all images are oriented correctly,
        regardless of how the image was captured
    -   Resize (Stretch to 640x640): YOLOv8 requires input images to be
        of a consistent size while maintaining a balance between speed
        and accuracy
-   Augmentations (per each image there will be output three ones)
    account for respectively
    -   Flip (Horizontal): different angles and perspectives
    -   Crop (0% Minimum Zoom, 15% Maximum Zoom): different distances
    -   Rotation (Between -10° and +10°): mis-alignement with camera
        angle
    -   Shear (±2° Horizontal, ±2° Vertical): different viewing
        conditions
    -   Grayscale (Apply to 10% of images) and Hue, Saturation,
        Brightness, Exposure (Between -15° and +15%, Between -15% and
        +15%): low-light conditions and color variations
    -   Blur (Up to 0.5px): minor focus issues
    -   Cutout (5 boxes with 2% size each): occlusion by randomly
        removing parts of the images

These augmentations ensure that the model can accurately detect and
recognize license plates under a wide range of environmental and
operational conditions, thereby enhancing the robustness and reliability
of the pipeline.

Weights obtained at the conclusion of each epoch during training have
been made available in the folder 'runs2/train/my_model'. However, only
the best-performing weights, based on validation accuracy and loss, have
been employed in the final model. This selective approach ensures
optimal performance by utilizing the most effective version of the
model, reducing errors and improving accuracy in vehicle and license
plate detection.

## Pipeline

1.  Load Models  
2.  Load Video and Read Frames The video is loaded and processed frame
    by frame. This method ensures that dynamic changes within the video
    are captured promptly.
3.  Detect and Track Vehicles Vehicles are detected to locate the region
    of interest (RoI) where the license plate is likely situated.
    Vehicles identifiers are tracked across frames due to the potential
    issue of non-consecutive frame detection. Tracking allows for
    interpolating frames where plate data might be missing, ensuring
    continuity.  
4.  Detect License Plates
5.  Assign License Plate to Vehicle The system verifies if a detected
    license plate belongs to an already detected vehicle. This tracking
    is essential, since solution frames lacking of information are
    interpolated and later saved in 'result_interpolated.csv' to
    maintain data integrity.
6.  Crop and Process License Plate Once the license plate is detected it
    undergoes grayscale conversion and thresholding. These steps
    highlight the text on the license plate, preparing it for more
    accurate character recognition.
7.  Read License Plate Number The processed image of the license plate
    is checked against the French license plate format. EasyOCR reads
    the text, adjusting for potential character misinterpretations e.g.
    digit-char mismatches. The system outputs the license plate's text
    and a confidence score.  
    Detection confidence scores are retained to manage multiple
    potential readings from the same plate, ensuring the most accurate
    text is chosen based on the highest confidence score. The results
    are saved in 'result.csv'.
8.  Save Results
