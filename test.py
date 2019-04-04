import cv2, os, random
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image
from keras import optimizers
from keras import backend as k
import numpy as np


def load_model(width, height):
    k.clear_session()
    json_file_path = 'xception_424_150x150.json'
    weights_path = 'xception_424_150x150.h5'

    # Model reconstruction from JSON file
    with open(json_file_path, 'r') as f:
        model = model_from_json(f.read()) 

    # Load weights into the new model
    model.load_weights(weights_path)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(amsgrad=True),
        metrics=['acc'])
    return model

def detect_crop_face(image, faceCascade):
    face_cascade = cv2.CascadeClassifier(faceCascade)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_face_size = []
    for (x,y,w,h) in faces:
        # squared_face =  cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
        cropped_face_size.append([x, y, w, h])
    return cropped_face_size #cropped_face = image[y:y+h, x:x+w]

def name_loader(index):
    if index == 0:
        return "Julie", '/Users/xingdawang/Movies/julie'
    elif index == 1:
        return "Melanie", '/Users/xingdawang/Movies/melanie'
    elif index == 2:
        return "Xingda", '/Users/xingdawang/Movies/xingda'

def get_euclidean_distance(source, target):
    euclidean_distance = source - target
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def random_image_loader(images_folder):
    # randomly select a image from given image folder
    random_file_name = random.choice(os.listdir(images_folder))
    image_path = os.path.join(images_folder, random_file_name)
    return image_path

def interactive(width, height, faceCascade = 'haarcascade_frontalface_default.xml'):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("loading")
    img_counter = 0
    # load face recogition model
    model = load_model(width=width, height=height)

    while True:
        ret, frame = cam.read()
        # get cropped face size parameters
        cropped_face_size = detect_crop_face(frame, faceCascade)
        # if face is found, iterate all found faces
        if cropped_face_size is not None:
            for size in cropped_face_size:
                # get face size
                x, y, w, h = size
                frame =  cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
                # predict face based on model
                cropped_face = frame[y:y+h, x:x+w]
                cropped_face_raw = cv2.resize(cropped_face,(width,height))
                cropped_face_raw = image.img_to_array(cropped_face_raw)
                cropped_face_raw = np.expand_dims(cropped_face_raw, axis=0)
                cropped_face_raw /= 255.
                # predict person
                classes = model.predict_classes(cropped_face_raw)
                # predicted percentage
                people = model.predict(cropped_face_raw).reshape(-1)
                name, path = name_loader(classes[0])

                # calculate enclidean distance
                image_path = random_image_loader(path)
                img = cv2.imread(image_path)
                img_resized = cv2.resize(img, (width, height))
                img_reshaped = img_resized.reshape(3, -1)

                cropped_face_resized = cv2.resize(cropped_face, (width, height))
                cropped_face_reshaped = image.img_to_array(cropped_face_resized).reshape(3, -1)
                result = get_euclidean_distance(img_reshaped, cropped_face_reshaped)
                # print(str(result) + " " + name)
                enclidean_distance_constant = 15000
                if result < enclidean_distance_constant:
                    # print name on model
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (x+10,y-10), font, 1,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow("frame", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "screenshot_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

width, height = 150, 150
interactive(width=width, height=height)