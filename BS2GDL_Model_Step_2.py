import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Flatten,
    Dense, concatenate, Layer, Activation
)
from bspline import BSplineLayer
from loss_functions import LCCLossLayer, DiceLossLayer

class AttentionGate(Layer):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        # Convolution pour traiter l'input venant du chemin de décoder
        self.W_g = Conv2D(F_g, kernel_size=1, padding='same', use_bias=True)
        # Convolution pour traiter l'input venant du chemin de l'encodeur
        self.W_x = Conv2D(F_l, kernel_size=1, padding='same', use_bias=False)
        # Convolution pour réduire le nombre de filtres pour la sortie de la porte d'attention
        self.psi = Conv2D(1, kernel_size=1, padding='same', use_bias=True, activation='sigmoid')
        # Activation relu
        self.relu = Activation('relu')
        # Convolution pour combiner Wg et Wx
        self.W = Conv2D(F_int, kernel_size=1, padding='same', use_bias=True)

    def call(self, g, x):
        # g: caractéristique du chemin de décodeur
        # x: caractéristique du chemin de l'encodeur
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.W(psi)
        psi = self.psi(psi)
        # Retourner x multiplié par la carte d'attention psi
        return x * psi


class CustomUNetModel(tf.keras.Model):
    def __init__(self, num_points=10, input_size=(256, 256, 1), bspline_points=20):
        super(CustomUNetModel, self).__init__()
        self.input_size = input_size
        self.num_points = num_points
        self.bspline_points = bspline_points
        self.lcc_loss_layer = LCCLossLayer()
        self.dice_loss_layer = DiceLossLayer()

        self.AttentionGate1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.AttentionGate2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.AttentionGate3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.AttentionGate4 = AttentionGate(F_g=32, F_l=32, F_int=16)

        # Architecture
        self.c1a = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', batch_input_shape=(None, 256, 256, 1))
        self.c1b = BatchNormalization()
        self.c1c = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c1d = BatchNormalization()
        self.p1 = MaxPooling2D((2, 2))
        self.d1 = Dropout(0.4)

        self.c2a = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c2b = BatchNormalization()
        self.c2c = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c2d = BatchNormalization()
        self.p2 = MaxPooling2D((2, 2))
        self.d2 = Dropout(0.45)

        self.c3a = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c3b = BatchNormalization()
        self.c3c = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c3d = BatchNormalization()
        self.p3 = MaxPooling2D((2, 2))
        self.d3 = Dropout(0.45)

        self.c4a = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c4b = BatchNormalization()
        self.c4c = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c4d = BatchNormalization()
        self.p4 = MaxPooling2D((2, 2))
        self.d4 = Dropout(0.45)

        # Partie du milieu
        self.c5a = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c5b = BatchNormalization()
        self.c5c = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c5d = BatchNormalization()
        self.d5 = Dropout(0.45)

        # Partie d'expansion
        self.u6 = UpSampling2D((2, 2))
        self.c6a = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c6b = BatchNormalization()
        self.c6c = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c6d = BatchNormalization()

        self.u7 = UpSampling2D((2, 2))
        self.c7a = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c7b = BatchNormalization()
        self.c7c = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c7d = BatchNormalization()

        self.u8 = UpSampling2D((2, 2))
        self.c8a = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c8b = BatchNormalization()
        self.c8c = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c8d = BatchNormalization()

        self.u9 = UpSampling2D((2, 2))
        self.c9a = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c9b = BatchNormalization()
        self.c9c = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.c9d = BatchNormalization()

        # Partie de régression pour prédire les points nodaux
        self.reg1 = Conv2D(128, (1, 1), activation='relu', padding='same')
        self.reg2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.reg3 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.reg4 = Conv2D(64, (3, 3), activation='relu', padding='same')

        self.flatten = Flatten()
        self.dense_1 = Dense(256, activation='relu')
        self.dense_2 = Dense(128, activation='relu')
        self.dense_3 = Dense(64, activation='relu')
        self.dense_4 = Dense(32, activation='relu')

        # Couche B-spline (doit être définie par vous)
        self.bspline_layer = BSplineLayer(bspline_points)

        self.output_nodal_layer = Dense(num_points * 2, activation='sigmoid', name='output_nodal')
        self.output_dist_layer = Conv2D(2, (1, 1), padding='same', activation='linear', name='output_dist')

        # Ajout d'une couche Lambda pour multiplier les sorties par 255
        # self.scale_layer = Lambda(lambda x: x * 255)

    def call(self, inputs, training=True):
        c1 = self.c1a(inputs)
        c1 = self.c1b(c1)
        c1 = self.c1c(c1)
        c1 = self.c1d(c1)
        p1 = self.p1(c1)
        d1 = self.d1(p1)

        c2 = self.c2a(d1)
        c2 = self.c2b(c2)
        c2 = self.c2c(c2)
        c2 = self.c2d(c2)
        p2 = self.p2(c2)
        d2 = self.d2(p2)

        c3 = self.c3a(d2)
        c3 = self.c3b(c3)
        c3 = self.c3c(c3)
        c3 = self.c3d(c3)
        p3 = self.p3(c3)
        d3 = self.d3(p3)

        c4 = self.c4a(d3)
        c4 = self.c4b(c4)
        c4 = self.c4c(c4)
        c4 = self.c4d(c4)
        p4 = self.p4(c4)
        d4 = self.d4(p4)

        # Partie du milieu
        c5 = self.c5a(d4)
        c5 = self.c5b(c5)
        c5 = self.c5c(c5)
        c5 = self.c5d(c5)
        d5 = self.d5(c5)

        # Partie d'expansion
        u6 = self.u6(d5)
        c4 = self.AttentionGate1(c4, u6)
        u6 = concatenate([u6, c4])
        x = self.c6a(u6)
        x = self.c6b(x)
        x = self.c6c(x)
        x = self.c6d(x)

        u7 = self.u7(x)
        c3 = self.AttentionGate2(c3, u7)  # Second attention gate, assumed to be defined in __init__
        u7 = concatenate([u7, c3])
        x = self.c7a(u7)
        x = self.c7b(x)
        x = self.c7c(x)
        x = self.c7d(x)

        u8 = self.u8(x)
        c2 = self.AttentionGate3(c2, u8)  # Third attention gate
        u8 = concatenate([u8, c2])
        x = self.c8a(u8)
        x = self.c8b(x)
        x = self.c8c(x)
        x = self.c8d(x)

        u9 = self.u9(x)
        c1 = self.AttentionGate4(c1, u9)  # Fourth attention gate
        u9 = concatenate([u9, c1])
        x = self.c9a(u9)
        x = self.c9b(x)
        x = self.c9c(x)
        x = self.c9d(x)

        reg_output = self.reg1(c5)
        reg_output = self.reg2(reg_output)
        #reg_output = self.reg3(reg_output)
        #reg_output = self.reg4(reg_output)
        reg_output = self.flatten(reg_output)
        reg_output = self.dense_1(reg_output)
        reg_output = self.dense_2(reg_output)
        #reg_output = self.dense_3(reg_output)
        #reg_output = self.dense_4(reg_output)

        output_dist = self.output_dist_layer(x)

        output_nodal = self.output_nodal_layer(reg_output)

        # Appliquer la multiplication par 255 ici
        # output_nodal_scaled = self.scale_layer(output_nodal)

        reshape_nodal = tf.reshape(output_nodal, [tf.shape(x)[0], self.num_points, 2])

        bspline_coord_pred = self.bspline_layer(reshape_nodal)

        # lcc loss function between nodal points and signed distance map
        _ = self.lcc_loss_layer([bspline_coord_pred, output_dist])

        # dice loss between Masks true and predicted
        _ = self.dice_loss_layer([bspline_coord_pred, output_dist])

        return {'output_dist': output_dist, 'output_nodal': reshape_nodal}
        # return {'output_dist': output_dist, 'output_nodal': reshape_nodal, 'lcc_loss': lcc_loss_value}

    def generate_bspline_points(self, output_nodal):
        # Redimensionner output_nodal pour qu'il ait la forme [-1, self.num_points, 2]
        reshaped_output_nodal = tf.reshape(output_nodal, [-1, self.num_points, 2])

        # Utiliser self.bspline_layer pour générer les points B-spline
        bspline_coord = self.bspline_layer(reshaped_output_nodal)

        bspline_curve_x, bspline_curve_y = tf.split(bspline_coord, num_or_size_splits=2, axis=-1)

        bspline_curve_x = abs(bspline_curve_x)
        bspline_curve_y = abs(bspline_curve_y)

        return bspline_curve_x, bspline_curve_y

    @property
    def metrics(self):
        # Add equidistant_loss_metric to the model's metrics
        base_metrics = super(CustomUNetModel, self).metrics
        return base_metrics + [self.lcc_loss_layer.lcc_loss_metric, self.dice_loss_layer.dice_loss_metric]  #

