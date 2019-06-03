from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import cv2


def main():
    model = load_model('./model_dog_cat.h5')

    # img_path = './gaijiinu.png'
    # img = img_to_array(load_img(img_path, target_size=(32, 32)))
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = frame[:, 80:560]
        cv2.imshow('frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        resized = cv2.resize(frame, (32, 32))
        reshaped = resized.reshape([1, 32, 32, 3])
        label = ['Cat', 'Dog']
        pred = model.predict(reshaped, batch_size=1, verbose=0)
        score = np.max(pred)
        pred_label = label[np.argmax(pred[0])]
        print('name:', pred_label)
        print('score:', score)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()