import argparse
import time
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters.detect import get_objects

def draw_boxes(frame, objs, labels, scale_x, scale_y):
    for obj in objs:
        bbox = obj.bbox
        xmin = int(bbox.xmin * scale_x)
        ymin = int(bbox.ymin * scale_y)
        xmax = int(bbox.xmax * scale_x)
        ymax = int(bbox.ymax * scale_y)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = labels.get(obj.id, obj.id)
        cv2.putText(frame, f'{label} {obj.score:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default='hand_model_edgetpu.tflite')
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} model'.format(args.model))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    labels = {0: 'Hand'}  # Adjust this based on your model's labels

    inference_size = common.input_size(interpreter)
    print(f"Model input size: {inference_size[0]}x{inference_size[1]}")

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = []
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess the frame
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        
        # Run inference
        start_time = time.monotonic()
        common.set_input(interpreter, cv2_im_rgb)
        interpreter.invoke()
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        inference_time = time.monotonic() - start_time

        # Calculate scaling factors
        scale_x, scale_y = frame.shape[1] / inference_size[0], frame.shape[0] / inference_size[1]

        # Draw bounding boxes on the original frame
        draw_boxes(frame, objs, labels, scale_x, scale_y)

        # Calculate and display FPS
        fps_counter.append(1.0 / (time.monotonic() - start_time))
        fps = sum(fps_counter[-30:]) / min(30, len(fps_counter))

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"FPS: {fps:.2f}, Inference time: {inference_time:.3f} seconds")

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
