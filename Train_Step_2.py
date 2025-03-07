import tensorflow as tf
from data_loader import load_data
from model import CustomUNetModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Define paths (generalized)
BASE_DIR = '/home/mofakhfa/Mohamed/Dataset_2'
OUTPUT_DIR = '/home/mofakhfa/Mohamed/Results/Bspline_CNN/Dataset_2/L_1234'

IMAGE_FOLDER = os.path.join(BASE_DIR, 'Seg/train/frames_all')
MASK_EPI_FOLDER = os.path.join(BASE_DIR, 'Dist/train/dist_endo/')
CONT_EPI_FOLDER = os.path.join(BASE_DIR, 'Contours/train/contours_endo/')
JSON_FILE = os.path.join(BASE_DIR, 'train_nodal_points_endo_40.json')
SEG_FOLDER = os.path.join(BASE_DIR, 'Seg/train/masks_endo')

def RMSE_loss():
    """
    Root Mean Square Error (RMSE) loss for regression tasks.
    """
    def loss(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    return loss

# Load data
images, masks_epi, nodal_points, contours_epi, seg_epi = load_data(
    IMAGE_FOLDER, MASK_EPI_FOLDER, CONT_EPI_FOLDER, SEG_FOLDER, JSON_FILE
)

# Normalize data
images /= 255.0
masks_epi /= 255.0
contours_epi /= 255.0
seg_epi /= 255.0

# Create background masks
mask_background = np.logical_not(masks_epi.astype(bool)).astype(int)
mask_background_seg = np.logical_not(seg_epi.astype(bool)).astype(int)

# Combine masks
masks = np.concatenate([masks_epi, mask_background], axis=-1)
segmentation_epi = np.concatenate([seg_epi, mask_background_seg], axis=-1)

# Merge data
combined_data = list(zip(images, masks, nodal_points, contours_epi, segmentation_epi))

# Split into train and test sets
train, test = train_test_split(combined_data, test_size=0.25, random_state=42)

# Unpack data
X_train, y_train_masks, y_train_nodal, y_train_contours, y_train_seg = zip(*train)
X_test, y_test_masks, y_test_nodal, y_test_contours, y_test_seg = zip(*test)

# Convert to NumPy arrays
X_train = np.array(X_train)
y_train_masks = np.array(y_train_masks)
y_train_nodal = np.array(y_train_nodal)
y_train_contours = np.array(y_train_contours)
y_train_seg = np.array(y_train_seg)
X_test = np.array(X_test)
y_test_masks = np.array(y_test_masks)
y_test_nodal = np.array(y_test_nodal)
y_test_contours = np.array(y_test_contours)
y_test_seg = np.array(y_test_seg)

print("X_train", tf.shape(X_train))
print("y_train_masks", tf.shape(y_train_masks))
print("y_train_nodal", tf.shape(y_train_nodal))
print("X_test", tf.shape(X_test))
print("y_test_masks", tf.shape(y_test_masks))
print("y_test_nodal", tf.shape(y_test_nodal))
print("y_test_seg", tf.shape(y_test_seg))
print("----------------------------------")

# Initialize model
model = CustomUNetModel(num_points=40, input_size=(256, 256, 1), bspline_points=20)

model.load_weights('model_weights_step_1.h5')

# Après avoir chargé les poids, vous pouvez créer le modèle fonctionnel si nécessaire
functional_model = Model(inputs=inputs, outputs=outputs)


def rmsle_loss_():
    def loss(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))
    return loss


# functional_model.compile(
#     optimizer='adam',
#     loss={
#         'output_dist': 'mean_squared_error',
#         'output_nodal': rmsle_loss_(),
#     },
#     loss_weights={
#         'output_dist': 1.0,
#         'output_nodal': 1.0
#     }
#
# )
# """
#     metrics={
#         'output_dist': ['mae'],
#         'output_nodal': ['mae']
#     }
# """


def custom_loss_function(y_true, y_pred, model):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    rmsle_loss = rmsle_loss_(y_true, y_pred)

    lcc_loss = model.lcc_loss_layer.lcc_loss_value
    dice_loss = model.equidistant_loss_layer.equidistant_loss_value

    total_loss = (mse_loss_weight * mse_loss +
                  combined_loss_weight * rmsle_loss +
                  lcc_loss_weight * lcc_loss +
                  dice_loss_weight * dice_loss)

    return total_loss

functional_model.compile(optimizer=Adam(lr=1e-3),
              loss=lambda y_true, y_pred: custom_loss_function(y_true, y_pred, model),
              metrics=['mae'])


# Création de l'Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# history = functional_model.fit(
#     x=X_train,
#     y={'output_dist': y_train_masks, 'output_nodal': y_train_nodal},
#     batch_size=32,
#     validation_data=(X_test, {'output_dist': y_test_masks, 'output_nodal': y_test_nodal}),
#     epochs=300,
#     callbacks=[early_stopping, reduce_lr]
# )


history = functional_model.fit(
    [X_train, y_train_nodal],
    [y_train_masks, y_train_nodal],
    batch_size=32,
    validation_data=([X_test, y_test_nodal], [y_test_masks, y_test_nodal]),
    epochs=300,
    callbacks=[early_stopping, reduce_lr]
)