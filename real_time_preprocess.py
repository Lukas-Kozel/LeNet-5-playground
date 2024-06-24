# this code is expecting number to be on white paper with nothing else for simplicity
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite('real_time_testing/step1_binary_image.jpg', binary_image)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    #binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    #binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    if np.mean(binary_image) > 127:  # Assuming black numbers on white background
        binary_image = cv2.bitwise_not(binary_image)
    
    cv2.imwrite('real_time_testing/step2_morph_image.jpg', binary_image)
    return frame, binary_image
    #_, bin = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #return frame, bin

def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours based on contour area
    min_contour_area = 60
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if contours:
        return max(contours, key=cv2.contourArea)
    else:
        return None

def crop_image(image, contour):
    if contour is not None:
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box slightly to ensure the entire number is captured
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, image.shape[1] - x)
        h = min(h + 2 * margin, image.shape[0] - y)
        
        cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite('real_time_testing/cropped_image.jpg', cropped_image)
        return cropped_image
    else: return None


def maintain_grayscale(image):
    for row_idx, row in enumerate(image):
        for col_idx, pixel in enumerate(row):
            if pixel > 0.4:
                image[row_idx, col_idx] = 1.0
            elif pixel < 0.3:
                image[row_idx, col_idx] = 0.0

    return image


def find_the_number(frame) -> np.ndarray:
    cv2.imwrite('real_time_testing/not_preprocessed_image.jpg', frame)
    # Preprocess the image
    image, binary_image = preprocess_image(frame)
    
    # Find the largest contour
    contour = find_largest_contour(binary_image)
    
    # Crop the image
    cropped_image = crop_image(image, contour)
    
    if cropped_image is None or cropped_image.size == 0:
        return None
    
    # Resize the cropped image
    resized_cropped_image = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('real_time_testing/resized_cropped_image.jpg', resized_cropped_image)
    
    # Convert to grayscale if not already (needed for single channel input)
    if len(resized_cropped_image.shape) == 3:
        resized_cropped_image = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('real_time_testing/grayscale_resized_cropped_image.jpg', resized_cropped_image)
    
    # Normalize the image
    resized_cropped_image = resized_cropped_image / 255.0
    #resized_cropped_image = maintain_grayscale(resized_cropped_image)
    cv2.imwrite('real_time_testing/normalized_resized_cropped_image.jpg', (resized_cropped_image * 255).astype(np.uint8))
    
    # Convert to a TensorFlow tensor and add batch dimension
    tensor_image = tf.convert_to_tensor(resized_cropped_image, dtype=tf.float32)
    tensor_image = tf.expand_dims(tensor_image, axis=0)  # Add batch dimension (1, 32, 32)
    tensor_image = tf.expand_dims(tensor_image, axis=-1)  # Add channel dimension (1, 32, 32, 1)
    
    # Save or display the preprocessed image
    cv2.imwrite('real_time_testing/preprocessed_image.jpg', (resized_cropped_image * 255).astype(np.uint8))
    return tensor_image


def classify_number(image):
    if image is None:
        return "No number found", "0.0"
    
    model = load_model('lenet5_model.keras')
    predictions = model.predict(image)
    class_prediction = np.argmax(predictions, axis=1)
    predicted_index = class_prediction[0]
    
    # Get the probability of the predicted class
    prediction_probability = predictions[0][predicted_index]
    
    return str(predicted_index), str(prediction_probability)