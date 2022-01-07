import glob
import os.path

import cv2.cv2 as cv2
from tensorflow.python.keras.models import load_model

from utils import generate_bounding_box

model = "models/eardetector-unet-vgg16"
input_shape = (256, 256, 3)
model = load_model(model)

result_files_folder = "data/results"
test_files_folder = "data/test"
for image_file in sorted(glob.glob(f"{test_files_folder}/*.bmp")):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

    prediction = model.predict(image)
    x, y, w, h = generate_bounding_box(prediction)
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    file_name = os.path.basename(image_file)
    cv2.imwrite(os.path.join(result_files_folder, file_name))
