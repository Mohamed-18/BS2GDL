import tensorflow as tf
import numpy as np


class BSplineDetector:
    """
    Detector for calculating B-spline control points and evaluation.
    """

    def generate_circle_points(self, radius, num_points):
        theta = tf.linspace(0.0, 2 * np.pi, num_points)
        x = radius * tf.cos(theta)
        y = radius * tf.sin(theta)
        return tf.stack([x, y], axis=1)


    def createPiToQi_tf(self, batch_radiuses):
        # batch_radiuses a une forme [batch_size, num_points]
        batch_size = tf.shape(batch_radiuses)[0]

        def process_single_radius_set(radiuses):
            radiuses = radiuses * 255
            nbNodal = tf.shape(radiuses)[0]
            Q = tf.zeros_like(radiuses, dtype=tf.float32)
            z1 = -2 + tf.sqrt(3.0)
            tn = nbNodal - 1
            reversed_indices = tf.range(nbNodal - 1, -1, -1, dtype=tf.float32)
            sommeR = tf.reduce_sum(
                (z1 ** reversed_indices) * tf.gather(tf.reshape(radiuses, [-1]), tf.cast(reversed_indices, tf.int32)))
            factor = 1.0 / (1.0 - z1 ** tf.cast(nbNodal, tf.float32))
            QTilde = tf.zeros_like(radiuses, dtype=tf.float32)
            update_value = tf.expand_dims([factor * sommeR], axis=-1)  # Cela donne à update_value la forme [1, 1]
            QTilde = tf.tensor_scatter_nd_update(QTilde, [[0]], update_value)
            for i in tf.range(1, nbNodal):
                z1_qtilde_prev = z1 * QTilde[i - 1]
                radius_current = radiuses[i]
                update_value = z1_qtilde_prev + radius_current
                QTilde = tf.tensor_scatter_nd_update(QTilde, [[i]], [update_value])

            sommeR = tf.reduce_sum(
                (z1 ** reversed_indices) * tf.gather(tf.reshape(QTilde, [-1]), tf.cast(reversed_indices, tf.int32)))
            factor = -(6.0 * z1 / (1.0 - z1 ** tf.cast(nbNodal, tf.float32)))
            update_value = [factor * sommeR]
            update_value = tf.reshape(update_value, [1, 1])  # Reshape pour correspondre à la forme [1, 1]
            Q = tf.tensor_scatter_nd_update(Q, [[0]], update_value)
            Q = tf.tensor_scatter_nd_update(Q, [[tn]], [z1 * Q[0] - 6 * z1 * QTilde[tn]])
            for i in range(nbNodal - 2, 0, -1):
                Q = tf.tensor_scatter_nd_update(Q, [[i]], [z1 * Q[i + 1] - 6 * z1 * QTilde[i]])
            Q = Q / 255
            return Q

        Q_batch_results = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for b in tf.range(batch_size):
            Q_b = process_single_radius_set(batch_radiuses[b])
            Q_batch_results = Q_batch_results.write(b, Q_b)

        Q_batch = Q_batch_results.stack()  # de forme [batch_size, num_points]

        return Q_batch

    def evaluate_bspline(self, s, Q):
        # Convertir 's' en float32 si ce n'est pas déjà le cas
        s = tf.cast(s, tf.float32)

        # Assurez-vous que Q est également de type float32
        Q = tf.cast(Q, tf.float32)

        term1 = (-1 / 6 * Q[0] + 1 / 2 * Q[1] - 1 / 2 * Q[2] + 1 / 6 * Q[3]) * s ** 3
        term2 = (1 / 2 * Q[0] - Q[1] + 1 / 2 * Q[2]) * s ** 2
        term3 = (-1 / 2 * Q[0] + 1 / 2 * Q[2]) * s
        term4 = 1 / 6 * Q[0] + 2 / 3 * Q[1] + 1 / 6 * Q[2]

        return term1 + term2 + term3 + term4

    def evaluate_bspline_closed(self, Qx, Qy, num_points=160):
        batch_size, n, _ = tf.shape(Qx)[0], Qx.shape[1] - 1, Qx.shape[2]
        s_values = tf.linspace(0.0, 1.0, 20)

        def evaluate_curve_segment(Qx, Qy, i):
            indices = (tf.range(i, i + 4) % n)
            segment_x = self.evaluate_bspline(s_values, tf.gather(Qx, indices, axis=0))
            segment_y = self.evaluate_bspline(s_values, tf.gather(Qy, indices, axis=0))
            return segment_x, segment_y

        def process_batch_element(Qx_b, Qy_b):
            curve_x = tf.TensorArray(tf.float32, size=n)
            curve_y = tf.TensorArray(tf.float32, size=n)
            for i in tf.range(n):
                seg_x, seg_y = evaluate_curve_segment(Qx_b, Qy_b, i)
                curve_x = curve_x.write(i, seg_x)
                curve_y = curve_y.write(i, seg_y)
            return curve_x.concat(), curve_y.concat()

        # Préparer les tensors pour recueillir les résultats
        curve_x_results = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        curve_y_results = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)

        # Process each batch element
        for b in tf.range(batch_size):
            curve_x_b, curve_y_b = process_batch_element(Qx[b], Qy[b])
            curve_x_results = curve_x_results.write(b, curve_x_b)
            curve_y_results = curve_y_results.write(b, curve_y_b)

        # Stack pour obtenir la forme finale [batch_size, 160, 1]
        curve_x_batch = curve_x_results.stack()
        curve_y_batch = curve_y_results.stack()

        return tf.expand_dims(curve_x_batch, -1), tf.expand_dims(curve_y_batch, -1)



class BSplineLayer(tf.keras.layers.Layer):
    """
    Layer for generating B-spline curves from nodal points.
    """

    def __init__(self, num_points=40, image_size=256, **kwargs):
        super(BSplineLayer, self).__init__(**kwargs)
        self.num_points = num_points
        self.image_size = image_size
        self.detector = BSplineDetector()

    def call(self, inputs):
        # Split des points nodaux en Qx et Qy
        Qx, Qy = tf.split(inputs, num_or_size_splits=2, axis=-1)

        # Conversion des tableaux Numpy en tenseurs TensorFlow
        Qx_tf = tf.convert_to_tensor(Qx, dtype=tf.float32)
        Qy_tf = tf.convert_to_tensor(Qy, dtype=tf.float32)

        # Convertir en points de contrôle
        control_points_x = self.detector.createPiToQi_tf(Qx_tf)
        control_points_y = self.detector.createPiToQi_tf(Qy_tf)

        # Évaluer la courbe B-spline pour l'ensemble des points de contrôle
        bspline_curve_x, bspline_curve_y = self.detector.evaluate_bspline_closed(control_points_x, control_points_y,
                                                                                 num_points=self.num_points)

        bspline_curve = tf.stack([bspline_curve_x, bspline_curve_y], axis=-1)
        return bspline_curve
