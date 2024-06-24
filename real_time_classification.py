import cv2
from real_time_preprocess import find_the_number, classify_number

file_path = "real_time_testing/test.jpg"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    prediction, probability = classify_number(find_the_number(frame))

    # Display the result
    cv2.putText(frame, f'Prediction: {prediction}, probability: {probability*100} %', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Realtime Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()