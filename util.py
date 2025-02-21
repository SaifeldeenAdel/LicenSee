import easyocr
import cv2
import csv
import os
import re

reader = easyocr.Reader(['ar'])


def get_car(plate, trackers):
  plate_x1, plate_y1, plate_x2, plate_y2, conf, class_id = plate

  for track in trackers:
    car_x1, car_y1, car_x2, car_y2, track_id = track
    if plate_x1 >= car_x1 and plate_y1 >= car_y1 and plate_x2 <= car_x2 and plate_y2 <= car_y2:
      return car_x1, car_y1, car_x2, car_y2, track_id

  return -1, -1, -1, -1, -1

def read_license_plate(plate_crop):
  detections = reader.readtext(plate_crop)
  best_text = None
  max_score = 0

  arabic_letters = "ء-ي"  # Unicode range for Arabic letters
  arabic_numbers = "٠-٩"  # Unicode range for Arabic numerals

  for detection in detections:
    bbox, text, score = detection
    if score > max_score:
      max_score = score

      # Normalize text: Remove spaces and non-alphanumeric characters
      text = text.replace(" ", "")
      text = ''.join(e for e in text if e.isalnum())

      # Arabic license plate format: First 3 are letters, last 4 are numbers
      pattern = rf"^[{arabic_letters}]{{3}}[{arabic_numbers}]{{4}}$"
      if re.match(pattern, text):
          best_text = text  # Keep the valid text

  # Ensure the score is above a threshold and text is valid
  if not best_text or max_score < 0.3:
    return None, 0
  print(best_text)
  return best_text, max_score


def visualize(frame, car_bbox, plate_bbox, license_crop_thresh):
  car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)

  frame = cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 0, 255), 2)
  
  if plate_bbox is None:
    return frame
  
  plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_bbox)
  frame = cv2.rectangle(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (255, 0, 0), 2)
  # Resize for better visualization
  h, w = license_crop_thresh.shape
  overlay = cv2.resize(license_crop_thresh, (int(w * 1.5), int(h * 1.5)))

  overlay_x, overlay_y = car_x1, car_y1 - (h * 3) - 10  # Position above the car
  if overlay_y < 0:
      overlay_y = car_y2 + 10  # If out of bounds, place below the car

  # Ensure overlay fits within the frame
  oh, ow = overlay.shape
  if overlay_x + ow > frame.shape[1]:
      overlay_x = frame.shape[1] - ow
  if overlay_y + oh > frame.shape[0]:
      overlay_y = frame.shape[0] - oh

  # Add overlay to frame
  frame[overlay_y:overlay_y + oh, overlay_x:overlay_x + ow] = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

  return frame


# def save_results(results, filename):
#   # Delete results.csv if it already exists
#   try:
#     os.remove(filename)
#   except FileNotFoundError:
#     pass

#   with open("results.csv", mode="w", newline="",encoding="utf-8-sig") as file:
#     writer = csv.writer(file)

#     # Write the header row
#     writer.writerow(["frame_id", "car_id", "car", "plate",
#                   "plate_text", "score"])

#     # Iterate through results dictionary
#     for frame_id, frame_data in results.items():
#         for car_id, car_info in frame_data.items():
            
#             car_bbox = car_info["car"]["bounding_box"]
#             plate_bbox = car_info["license_plate"]["bbox"]
#             plate_text = car_info["license_plate"]["text"]
#             score = car_info["license_plate"]["score"]

#             # Write row data
#             writer.writerow([frame_id, 
#                             car_id,
#                             car_bbox, 
#                             plate_bbox, 
#                             plate_text, 
#                             score])
            
from collections import defaultdict

def get_best_plate(plate_detections, confidence_threshold=0.4):
    print(plate_detections)
    plate_counts = defaultdict(list)  # Stores scores for each plate text

    for plate_text, score in plate_detections:
        if score >= confidence_threshold:
            plate_counts[plate_text].append(score)  # Store confidence for each text

    if not plate_counts:
        return None, None  # No valid plates

    # Select the most frequent plate text (tie-breaker: highest average confidence)
    best_plate = max(plate_counts.items(), key=lambda x: (len(x[1]), sum(x[1]) / len(x[1])))
    
    return best_plate[0], sum(best_plate[1]) / len(best_plate[1])  # (plate_text, avg_score)

def save_results(results, filename):
    # Delete results.csv if it already exists
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    # Dictionary to store all detected plates for each car
    car_plates = defaultdict(list)

    # Collect plate detections per car
    for frame_id, frame_data in results.items():
        for car_id, car_info in frame_data.items():
            car_bbox = car_info["car"]["bounding_box"]
            plate_bbox = car_info["license_plate"]["bbox"]
            plate_text = car_info["license_plate"]["text"]
            score = car_info["license_plate"]["score"]
            car_plates[car_id].append((plate_text, score, frame_id, car_bbox, plate_bbox))

    with open(filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "car", "plate", "plate_text", "score"])

        # Process each car and select the best license plate text
        for car_id, detections in car_plates.items():
            plate_text, avg_score = get_best_plate([(text, score) for text, score, _, _, _ in detections])

            if plate_text:  # Only save if a valid plate was found
                # Use the first frame's bounding boxes (they are mostly consistent)
                frame_id, car_bbox, plate_bbox = next((f, cb, pb) for _, _, f, cb, pb in detections)

                writer.writerow([frame_id, car_bbox, plate_bbox, plate_text, avg_score])