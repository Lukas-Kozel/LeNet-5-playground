# this code is expecting number to be on white paper with nothing else for simplicity
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    _, bin = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image, bin

def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return max(contours,key=cv2.contourArea)

def crop_image(image, contour):
    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image


def maintain_grayscale():
    #TODO
    return


def find_the_number(image_path) -> np.ndarray:
    # Preprocess the image
    image, binary_image = preprocess_image(image_path)
    
    # Find the largest contour
    contour = find_largest_contour(binary_image)
    
    # Crop the image
    cropped_image = crop_image(image, contour)
    
    resized_cropped_image = cv2.resize(cropped_image,(32,32), interpolation=cv2.INTER_LINEAR)
        # Convert to grayscale if not already (needed for single channel input)
    if len(resized_cropped_image.shape) == 3:
        resized_cropped_image = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Convert to a TensorFlow tensor and add batch dimension
    tensor_image = tf.convert_to_tensor(resized_cropped_image, dtype=tf.float32)
    tensor_image = tf.expand_dims(tensor_image, axis=0)  # Add batch dimension (1, 32, 32)
    
    # Save or display the cropped image
    cv2.imwrite('preprocessed_image.jpg', resized_cropped_image)
    #cv2.imshow('Cropped Image', resized_cropped_image)
    return tensor_image


def classify_number(image):
    # Load the entire model back from the file
    model = load_model('../lenet5_model.keras')
    model.summary()
    predictions = model.predict(image)
    class_prediction = np.argmax(predictions, axis=1)
    print(f'Predicted class: {class_prediction}')
    return str(class_prediction[0])


# Example usage
#image_path = 'uploaded_images/5.jpg'
#main(image_path)