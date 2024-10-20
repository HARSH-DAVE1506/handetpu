import time
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect

def draw_boxes(frame, detections, scale_x, scale_y):
    for obj in detections:
        bbox = obj.bbox
        xmin = int(bbox.xmin * scale_x)
        ymin = int(bbox.ymin * scale_y)
        xmax = int(bbox.xmax * scale_x)
        ymax = int(bbox.ymax * scale_y)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{obj.id} {obj.score:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    model_path = 'hand_model_edgetpu.tflite'

    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    _, height, width, _ = input_details[0]['shape']
    
    print(f"Model input size: {width}x{height}")

    cap = cv2.VideoCapture(0)
    
    # Set a lower resolution for processing
    process_width, process_height = 320, 240
    
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Force resize to lower resolution
        frame_small = cv2.resize(frame, (process_width, process_height))

        # Prepare input tensor
        input_tensor = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (width, height))

        # Run inference
        start_time = time.time()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        detections = detect.get_objects(interpreter, score_threshold=0.3, top_k=5)
        inference_time = time.time() - start_time

        # Calculate scaling factors
        scale_x = frame.shape[1] / width
        scale_y = frame.shape[0] / height

        # Draw bounding boxes on the original frame
        draw_boxes(frame, detections, scale_x, scale_y)

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
