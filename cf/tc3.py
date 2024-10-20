import time
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image

def draw_boxes(frame, detections):
    for obj in detections:
        bbox = obj.bbox
        cv2.rectangle(frame, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f'{obj.id} {obj.score:.2f}', (int(bbox.xmin), int(bbox.ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    model_path = 'hand_model_edgetpu.tflite'

    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    _, height, width, _ = input_details[0]['shape']
    
    print(f"Model input size: {width}x{height}")

    cap = cv2.VideoCapture(0)
    # Set lower resolution for the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Prepare the input tensor once
    input_tensor = np.zeros((height, width, 3), dtype=np.uint8)

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess the frame
        resized_frame = cv2.resize(frame, (width, height))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_tensor[:] = resized_frame

        # Run inference
        start_time = time.time()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        detections = detect.get_objects(interpreter, score_threshold=0.3)
        inference_time = time.time() - start_time

        # Draw bounding boxes
        draw_boxes(frame, detections)

        # Calculate and display FPS
        frame_count += 1
        if (time.time() - fps_time) > 1:
            fps = frame_count / (time.time() - fps_time)
            fps_time = time.time()
            frame_count = 0
            print(f"FPS: {fps:.2f}, Inference time: {inference_time:.3f} seconds")

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
