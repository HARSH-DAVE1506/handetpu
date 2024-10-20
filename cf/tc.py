import time
import cv2
from edgetpu.detection.engine import DetectionEngine

# Function to draw bounding boxes on the frame
def draw_boxes(frame, box, label, score):
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(frame, f'{label} {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Main function
def main():
    model_path = 'hand_model_edgetpu.tflite'  # Hardcoded model path

    # Initialize engine
    engine = DetectionEngine(model_path)

    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        start_time = time.time()
        ans = engine.DetectWithImage(rgb_frame, threshold=0.05, keep_aspect_ratio=True, relative_coord=True)
        end_time = time.time()

        # Draw bounding boxes
        for obj in ans:
            box = obj.bounding_box.flatten().tolist()
            box = [box[0] * frame.shape[1], box[1] * frame.shape[0], box[2] * frame.shape[1], box[3] * frame.shape[0]]
            draw_boxes(frame, box, 'Hand', obj.score)

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
