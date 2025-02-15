import easyocr
import cv2
import csv
import os

reader = easyocr.Reader(['en'])


def get_car(plate, trackers):
  plate_x1, plate_y1, plate_x2, plate_y2, conf, class_id = plate

  for track in trackers:
    car_x1, car_y1, car_x2, car_y2, track_id = track
    if plate_x1 >= car_x1 and plate_y1 >= car_y1 and plate_x2 <= car_x2 and plate_y2 <= car_y2:
      return car_x1, car_y1, car_x2, car_y2, track_id

  return -1, -1, -1, -1, -1

def read_license_plate(plate_crop):
  detections = reader.readtext(plate_crop)
  text = None
  score = 0
  for detection in detections:
    bbox, text, score = detection
    text = text.replace(" ", "")
    text = ''.join(e for e in text if e.isalnum())
    text = text.upper()
    if len(text) == 7 and score > 0.4:
      break

  if (text and len(text) != 7 )or score < 0.3:
    text = None
    score = 0
    
  return text, score


def visualize(frame, car_bbox, plate_bbox, license_crop_thresh):
  car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
  plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_bbox)

  frame = cv2.rectangle(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (255, 0, 0), 2)
  frame = cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 0, 255), 2)
  
  # Resize for better visualization
  h, w = license_crop_thresh.shape
  overlay = cv2.resize(license_crop_thresh, (w * 3, h * 3))

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


def save_results(results, filename):
  # Delete results.csv if it already exists
  try:
    os.remove(filename)
  except FileNotFoundError:
    pass

  with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["frame_id", "car", "plate",
                  "plate_text", "score"])

    # Iterate through results dictionary
    for frame_id, frame_data in results.items():
        for car_id, car_info in frame_data.items():
            car_bbox = car_info["car"]["bounding_box"]
            plate_bbox = car_info["license_plate"]["bbox"]
            plate_text = car_info["license_plate"]["text"]
            score = car_info["license_plate"]["score"]

            # Write row data
            writer.writerow([frame_id, 
                            car_bbox, 
                            plate_bbox, 
                            plate_text, 
                            score])