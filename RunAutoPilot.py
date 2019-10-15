import numpy as np
import cv2
import scipy.misc
from keras.models import load_model

model = load_model('Autopilot.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 60
    return steering_angle


def keras_process_image(img):
    image_x = 66
    image_y = 200
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 3))
    return img


steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture('run.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    image = scipy.misc.imresize(frame[-150:], [66, 200]) / 255.0
    steering_angle = keras_predict(model, image)
    print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle- smoothed_angle) / abs(
        steering_angle- smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
