import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #_, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    return image, binary_image

def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter out small contours based on contour area
    min_contour_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if contours:
        return max(contours, key=cv2.contourArea)
    else:
        return None

def crop_image(image, contour):
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        # Expand the bounding box slightly to ensure the entire number is captured
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, image.shape[1] - x)
        h = min(h + 2 * margin, image.shape[0] - y)
        
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return None

def find_the_number(image_path) -> np.ndarray:
    image, binary_image = preprocess_image(image_path)
    contour = find_largest_contour(binary_image)
    
    cropped_image = crop_image(image, contour)
    
    if cropped_image is None or cropped_image.size == 0:
        return None  # If no contour is found or the cropped image is empty, return None
    
    resized_cropped_image = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_LINEAR)
    
    if len(resized_cropped_image.shape) == 3:
        resized_cropped_image = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
    
    tensor_image = tf.convert_to_tensor(resized_cropped_image, dtype=tf.float32)
    tensor_image = tf.expand_dims(tensor_image, axis=0)  # Add batch dimension (1, 32, 32)
    
    cv2.imwrite('preprocessed_image.jpg', resized_cropped_image)
    return tensor_image

def classify_number(image):
    if image is None:
        return "No number found"
    
    model = load_model('../lenet5_model.keras')
    predictions = model.predict(image)
    class_prediction = np.argmax(predictions, axis=1)
    return str(class_prediction[0])

# Example usage with the provided image
image_path = '/home/luky/playground/handwriting_recognition/real_time_testing/real_time_test2.png'
tensor_image = find_the_number(image_path)
prediction = classify_number(tensor_image)
print(f'Prediction: {prediction}')
