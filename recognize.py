from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf

# Load pretrained model
pretrained_model = tf.keras.models.load_model("face_recognition_core_model.h5")

# Load MTCNN detector
detector = MTCNN()
def recognize_faces(video_capture, fps=50):
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert frame to grayscale for MTCNN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using MTCNN
        results = detector.detect_faces(frame)

        # Loop through each detected face and recognize
        for result in results:
            x, y, w, h = result['box']
            face = frame[y:y+h, x:x+w]

            # Preprocess face for prediction (resize, normalize, etc.)
            face = cv2.resize(face, (320, 320))  # Adjust size as per your model input shape
            face = face / 255.0  # Normalize pixel values

            # Perform face recognition using pretrained model
            prediction = pretrained_model.predict(np.expand_dims(face, axis=0))

            # Get predicted class (assuming one-hot encoding)
            predicted_class = np.argmax(prediction)
            
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text label with predicted class
            cv2.putText(frame, f'Class: {label_name[predicted_class]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check for key press
        key = cv2.waitKey(1000 // fps) & 0xFF  # Delay based on desired FPS

        # Break the loop when 'q' is pressed
        if key == ord('q') or key == 27:  # Check for 'q' or 'Esc' key
            break

    # Release the video capture object
    video_capture.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Run face recognition on webcam stream
video_capture = cv2.VideoCapture(0)
recognize_faces(video_capture)
