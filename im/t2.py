import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Load the model
model_path = 'hand_model_edgetpu.tflite'  # Hardcoded model path
label_path = 'custom_label.txt'  # Hardcoded label file path
image_path = 'timg.jpg'  # Hardcoded input image path

# Create an interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load the labels
labels = read_label_file(label_path)

# Load and preprocess the image
image = Image.open(image_path).convert('RGB')
image = image.resize((common.input_size(interpreter)), Image.ANTIALIAS)
input_data = np.array(image, dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)

# Set the input tensor
common.set_input(interpreter, input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = classify.get_output(interpreter)

# Display results
for i in range(len(output_data)):
    class_id = output_data[i][0].id
    score = output_data[i][0].score
    print(f'{labels[class_id]}: {score:.5f}')
