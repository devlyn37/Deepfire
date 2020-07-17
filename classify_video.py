import cv2
import math
import time
from cv2 import VideoCapture
from tensorflow import keras
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

LABELS = ["fire", "no fire"]
frames_per_second = 4

def get_image_generator(filename=None):

    # Setting parameter to 0 here will capture video from attached interface
    if filename:
        cap = cv2.VideoCapture(filename)
    else:
        cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)/frames_per_second

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id = cap.get(1)
        if (filename and frame_id % math.floor(fps) == 0):
            yield frame

        if not filename:
            before = time.time()
            yield frame
            remaining_time = max(0, 1 - (time.time() - before))
            time.sleep(remaining_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def resize(img):
    return cv2.resize(img, (224, 224)).reshape(1, 224, 224, 3)


def main():

    model = keras.models.load_model('saved_models/resnet50')

    #fp = open('framebyframe.txt', 'w')

    image_generator = get_image_generator('./aerial_video.mp4')
    # image_generator = get_image_generator()

    for frame in image_generator:
        res = model.predict(preprocess_input(resize(frame)))
        result_label = LABELS[res.argmax()]
        #fp.write(str(result_label)+"\n")
        print(f'{result_label} - {res[0]}')
        if "DISPLAY" in os.environ:
            cv2.imshow('frame', frame)
            while False:
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    break

    #fp.close()
if __name__ == '__main__':
    main()
