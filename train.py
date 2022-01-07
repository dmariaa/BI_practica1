import datetime
import os.path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from sklearn.model_selection import train_test_split

from model import build_vgg16_unet, build_unet
from dataset import generate_dataset

# load and split data
img_path = 'data/train/images'
mask_path = 'data/train/masks'
test_size = 0.04

X, Y_m, Y_a = generate_dataset(ear_img_path=img_path, ear_mask_path=mask_path, augm_size=7)
X_train, Y_m_train, Y_a_train, X_test, Y_m_test, Y_a_test = \
    train_test_split(X, Y_m, Y_a, test_size=test_size, random_state=1234, shuffle=True)

Y_m_train_1hot = to_categorical(Y_m_train)
Y_m_test_1hot = to_categorical(Y_m_test)

# build and compile model
input_shape = (256, 256, 3)
learning_rate = 1e-3

optimizer = Adam(learning_rate=learning_rate)
loss = CategoricalCrossentropy()

model = build_vgg16_unet(input_shape=input_shape, num_classes=2)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'FalsePositives'])

# launch training
batch_size = 16
epochs = 200
es = EarlyStopping

model_history = model.fit(X_train, Y_m_train_1hot, epochs=epochs, batch_size=batch_size,
                          validation_data=[X_test, Y_m_test_1hot], callbacks=[es])

# score = model.evaluate(X_test, Y_test_1hot)
# print(f"\n Presición con el conjunto de test: {score[2]*100:0.2f} %")

# here model folder can be setup with date or similar to save multiple models
model_folder = f"models/{datetime.date:%Y%m%d_%H:%M:%S}"
model_name = "eardetector-unet-vgg16.h5"
model.save(os.path.join(model_folder, model_name))