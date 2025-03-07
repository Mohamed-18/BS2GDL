import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean


"""""LCC Loss"""""
def lcc_loss(y_pred, distance_maps):
    loss = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # Utiliser un tableau pour accumuler les pertes

    batch_size = tf.shape(distance_maps)[0]
    width = 256
    for i in range(batch_size):
        distance_map = distance_maps[i] * 256
        y_pred_i = y_pred[i] * 256

        normalized_distance_map = tf.cast(distance_map, tf.float32)
        original_distance_map = (normalized_distance_map - 127) * 256 / 128

        # Redimensionner y_pred_i pour l'indexation
        y_pred_int = tf.cast(y_pred_i, tf.int32)
        y_pred_int = tf.reshape(y_pred_int, [-1, 2])

        flat_indices = y_pred_int[:, 0] + y_pred_int[:, 1] * width
        # tf.print("flat_indices", flat_indices)
        flat_indices = tf.clip_by_value(flat_indices, 0, width * width - 1)

        flat_distance_map = tf.reshape(original_distance_map, [-1])

        # Récupérer les distances interpolées
        interpolated_distances = tf.gather(flat_distance_map, flat_indices)

        lcc_loss_value = tf.reduce_mean(tf.square(interpolated_distances))

        # Ajouter la perte individuelle au tableau
        loss = loss.write(i, lcc_loss_value)

    # Empilez toutes les pertes individuelles dans un seul tenseur
    loss = loss.stack()

    # Calculez la moyenne de la perte sur l'ensemble du batch
    loss = tf.reduce_mean(loss)

    loss_batch = loss
    return loss_batch

class LCCLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LCCLossLayer, self).__init__(**kwargs)
        self.lcc_loss_metric = tf.keras.metrics.Mean(name='lcc_loss')
        self.lcc_loss_value = 0  # Initialisez à une valeur par défaut

    def call(self, y_pred):
        loss = lcc_loss(y_pred[0], y_pred[1])
        self.add_loss(loss)
        self.lcc_loss_value = loss  # Stocker la valeur de la perte
        self.lcc_loss_metric.update_state(loss)  # Mettre à jour la métrique avec la perte
        return y_pred



"""""Dice Loss"""""
class DiceLossLayer(tf.keras.layers.Layer):
    def __init__(self, loss_weight=1.0, **kwargs):
        super(DiceLossLayer, self).__init__(**kwargs)
        self.dice_loss_metric = tf.keras.metrics.Mean(name='dice_loss')
        self.dice_loss_value = 0

    @staticmethod
    def dice_loss(y_true, y_pred):
        smooth = 1e-6
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


    @staticmethod
    def fill_polygon(points_Bspline):
        shape = [256, 256]
        mask = tf.zeros(shape, dtype=tf.uint8)

        points = tf.clip_by_value(points_Bspline * 255, 0, 255)
        points = tf.squeeze(points, axis=1)

        min_y = tf.cast(tf.reduce_min(points[:, 1]), tf.int32)
        max_y = tf.cast(tf.reduce_max(points[:, 1]), tf.int32)

        def process_row(y, mask):
            nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for i in tf.range(tf.shape(points)[0]):
                j = i - 1 if i > 0 else -1
                if ((points[i, 1] < tf.cast(y, tf.float32) and points[j, 1] >= tf.cast(y, tf.float32)) or
                    (points[j, 1] < tf.cast(y, tf.float32) and points[i, 1] >= tf.cast(y, tf.float32))):
                    x = points[i, 0] + (tf.cast(y, tf.float32) - points[i, 1]) / (points[j, 1] - points[i, 1]) * (points[j, 0] - points[i, 0])
                    nodes = nodes.write(nodes.size(), x)

            nodes = nodes.stack()
            nodes = tf.sort(nodes)
            for i in tf.range(0, tf.shape(nodes)[0], 2):
                if i >= tf.shape(nodes)[0] - 1:
                    break
                x_start = tf.cast(nodes[i], tf.int32)
                x_end = tf.cast(nodes[i + 1], tf.int32)
                indices = tf.range(x_start, x_end + 1)
                indices = tf.stack([tf.fill([tf.shape(indices)[0]], y), indices], axis=1)
                updates = tf.ones(tf.shape(indices)[0], dtype=tf.uint8)
                mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            return mask

        y_range = tf.range(min_y, max_y + 1)
        for y in y_range:
            mask = process_row(y, mask)

        return mask


    @staticmethod
    def convert_distance_map_to_binary_mask(distance_map):
        distance_map = distance_map * 255
        seuil = 127
        binary_mask = tf.cast(distance_map <= seuil, tf.float32)
        return binary_mask

    def call(self, y_pred):
        batch_size = tf.shape(y_pred[1])[0]
        losses = tf.TensorArray(tf.float32, size=batch_size)

        for i in tf.range(batch_size):
            bspline_points = self.fill_polygon(y_pred[0][i])
            bspline_points = tf.cast(bspline_points, tf.float32)

            distance_map_mask = self.convert_distance_map_to_binary_mask(y_pred[1][i])
            distance_map_mask = distance_map_mask[..., 0]

            loss_dice = self.dice_loss(bspline_points, distance_map_mask)
            losses = losses.write(i, 1 - loss_dice)

        mean_loss_haus = tf.reduce_mean(losses.stack())
        self.add_loss(mean_loss_haus)
        self.dice_loss_value = mean_loss_haus
        self.dice_loss_metric.update_state(mean_loss_haus)

        return mean_loss_haus