from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, visualize, save_results


car_model = YOLO("yolo11n.pt")
license_plate_model = YOLO("license.pt")
tracker = Sort()
cap = cv2.VideoCapture("videos/egypt1.mp4")

results = {}
frame_count = 0
vehicles = [2,5,7]

while True:
  ret, frame = cap.read()
  if ret:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.resize(frame, (1280, 720))
    frame_count += 1
    # if frame_count > 60:
    #   break
    results[frame_count] = {}

    detections = car_model(frame)[0]
    cars = []
    for det in detections.boxes.data.tolist():
      # Find only vehicle detections
      x1, y1, x2, y2 , conf, class_id = det
      if int(class_id) in vehicles:
        cars.append([x1, y1, x2, y2, conf])

    # Track the vehicles
    cars = np.array(cars)
    print(len(cars))
    if len(cars) > 0:
      trackers = tracker.update(cars)

      # Detect License Plates
      plates = license_plate_model(frame)[0]
      
      for plate in plates.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = map(lambda x: round(x, 2), plate)
        car_x1, car_y1, car_x2, car_y2, car_id = map(lambda x: round(x, 2), get_car(plate, trackers))

        if car_id != -1:
          # Crop and Threshold the license plate
          license_crop = cv2.cvtColor(frame[int(y1): int(y2), int(x1): int(x2), :] , cv2.COLOR_BGR2GRAY)
          license_crop_thresh = cv2.threshold(license_crop, 100, 255, cv2.THRESH_BINARY_INV)[1]

          # Read the license plate
          plate_text, score = read_license_plate(license_crop_thresh)

          # Save the results
          if plate_text:
            results[frame_count][car_id] = {"car": 
                                              {"bounding_box": [car_x1, car_y1, car_x2, car_y2]}, 
                                            "license_plate": 
                                              {"bbox": [x1, y1, x2, y2], "text": plate_text, "score": score}}
            
          # Draw the bounding boxes
          frame = visualize(frame, (car_x1, car_y1, car_x2, car_y2), (x1, y1, x2, y2), license_crop_thresh)
          cv2.imshow("Frame", frame)
          # cv2.waitKey(0)
    else:
      cv2.imshow("Frame", frame)
  else:
    break
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
print(results)
save_results(results, "results.csv")

        


