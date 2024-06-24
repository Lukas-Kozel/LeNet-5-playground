import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite('real_time_testing/edges.jpg', edges)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and approximate contours
    min_contour_area = 100
    contours = [cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True) for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if contours:
        return frame, max(contours, key=cv2.contourArea)
    else:
        return frame, None

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
    else:
        return None

def find_the_number(frame) -> np.ndarray:
    cv2.imwrite('real_time_testing/not_preprocessed_image.jpg', frame)
    
    # Preprocess the image
    image, contour = preprocess_image(frame)
    
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