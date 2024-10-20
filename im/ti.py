import time
from PIL import Image
from PIL import ImageDraw
import numpy as np
from edgetpu.detection.engine import DetectionEngine

# Function to draw bounding boxes on the image
def draw_boxes(img, box, label, score):
    draw = ImageDraw.Draw(img)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red')
    draw.text((box[0], box[1]), f'{label} {score:.2f}', fill='red')

# Main function
def main():
    model_path = 'hand_model_edgetpu.tflite'  # Hardcoded model path
    input_image_path = 'timg.jpg'  # Hardcoded input image path

    # Initialize engine
    engine = DetectionEngine(model_path)

    # Load input image
    img = Image.open(input_image_path)

    # Convert image to RGB
    img = img.convert('RGB')

    # Run inference
    start_time = time.time()
    ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=True)
    end_time = time.time()

    # Draw bounding boxes
    for obj in ans:
        box = obj.bounding_box.flatten().tolist()
        box = [box[0] * img.width, box[1] * img.height, box[2] * img.width, box[3] * img.height]
        draw_boxes(img, box, 'Hand', obj.score)

    # Save output image
    img.save('output.jpg')

    # Print inference time
    print(f'Inference time: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    main()
