import time
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image

def draw_boxes(frame, box, label, score):
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.putText(frame, f'{label} {score:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    model_path = 'hand_model_edgetpu.tflite'

    # Initialize the TF Lite interpreter
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    print(f"Model input size: {width}x{height}")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (width, height))
        input_tensor = np.expand_dims(resized_frame, axis=0)

        # Run inference
        start_time = time.time()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        detections = detect.get_objects(interpreter, score_threshold=0.3)
        end_time = time.time()

        # Draw bounding boxes
        for obj in detections:
            bbox = obj.bbox
            # Scale bounding box to original frame size
            scale_x, scale_y = frame.shape[1] / width, frame.shape[0] / height
            bbox_scaled = [
                bbox.ymin * scale_y,
                bbox.xmin * scale_x,
                bbox.ymax * scale_y,
                bbox.xmax * scale_x
            ]
            draw_boxes(frame, bbox_scaled, obj.id, obj.score)

        # Display frame
        cv2.imshow('Hand Detection', frame)

        # Print inference time
        print(f'Inference time: {end_time - start_time:.2f} seconds')

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
