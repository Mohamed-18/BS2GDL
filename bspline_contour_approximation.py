import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from glob import glob
import json
json_data = {}

class BSplineDetector:
    def __init__(self):
        pass

    def generate_circle_points(self, radius, num_points):
        theta = tf.linspace(0.0, 2 * np.pi, num_points)
        x = radius * tf.cos(theta)
        y = radius * tf.sin(theta)
        return tf.stack([x, y], axis=1)

    def createPiToQi_tf(self, radiuses):
        nbNodal = tf.size(radiuses)
        Q = tf.zeros_like(radiuses, dtype=tf.float32)
        z1 = -2 + tf.sqrt(3.0)

        tn = nbNodal - 1

        reversed_indices = tf.range(nbNodal - 1, -1, -1, dtype=tf.float32)

        sommeR = tf.reduce_sum((z1 ** reversed_indices) * tf.gather(radiuses, tf.cast(reversed_indices, tf.int32)))
        factor = 1.0 / (1.0 - z1 ** tf.cast(nbNodal, tf.float32))
        QTilde = tf.Variable(tf.zeros_like(radiuses, dtype=tf.float32))

        QTilde = QTilde[0].assign(factor * sommeR)

        for i in tf.range(1, nbNodal):
            QTilde = QTilde[i].assign(z1 * QTilde[i - 1] + radiuses[i])

        sommeR = tf.reduce_sum((z1 ** tf.range(nbNodal, dtype=tf.float32)) * QTilde)
        factor = -(6.0 * z1 / (1.0 - z1 ** tf.cast(nbNodal, tf.float32)))
        Q = tf.tensor_scatter_nd_update(Q, [[0]], [factor * sommeR])

        Q = tf.tensor_scatter_nd_update(Q, [[tn]], [z1 * Q[0] - 6 * z1 * QTilde[tn]])

        for i in range(nbNodal - 2, 0, -1):
            Q = tf.tensor_scatter_nd_update(Q, [[i]], [z1 * Q[i + 1] - 6 * z1 * QTilde[i]])

        return Q

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

    def evaluate_bspline_closed(self, Qx, Qy, s_min=0.0, s_max=1.0, num_points=20):
        s_values = tf.linspace(s_min, s_max, num_points)
        curve_x, curve_y = [], []
        n = Qx.shape[0]

        for i in range(n):
            for s in s_values:
                indices = [(i + j) % n for j in range(4)]
                curve_x.append(self.evaluate_bspline(s, tf.gather(Qx, indices)))
                curve_y.append(self.evaluate_bspline(s, tf.gather(Qy, indices)))

        return tf.stack(curve_x), tf.stack(curve_y)


# ============================================================
# NonUniformBSplineFunctionRecursive2
# ============================================================
"""
Compute the value of a non-uniform B-spline basis function for a
given spline order, control point index, and parameter value.

This function implements the recursive Cox–de Boor formulation of
B-spline basis functions. The recursion decreases the spline order
at each call until reaching the base case (order 0).

Parameters
----------
knots : array-like
    Knot vector defining the parameter intervals over which the
    B-spline basis functions are defined.

order : int
    Order of the B-spline basis function. The spline order is
    equal to the polynomial degree plus one.

i : int
    Index of the control point associated with the basis function.

t : float
    Parameter value at which the basis function is evaluated.

Returns
-------
float
    Value of the B-spline basis function. This value represents
    the contribution (weight) of the i-th control point at the
    parameter location t.
"""


def NonUniformBSplineFunctionRecursive2(knots, order, i, t):
    if order == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1
        else:
            return 0
    else:
        numer1 = 0
        denom1 = knots[i + order] - knots[i]

        if denom1 != 0:
            numer1 = (t - knots[i]) * NonUniformBSplineFunctionRecursive2(knots, order - 1, i, t) / denom1

        numer2 = 0
        denom2 = knots[i + order + 1] - knots[i + 1]

        if denom2 != 0:
            numer2 = (knots[i + order + 1] - t) * NonUniformBSplineFunctionRecursive2(knots, order - 1, i + 1,
                                                                                      t) / denom2

        return numer1 + numer2


# ============================================================
# ComputeBendingMatrixCyclic
# ============================================================
"""
Compute the cyclic bending matrix used in the regularization term
of the B-spline fitting process.

The bending matrix is derived from the second-order derivatives of
the B-spline basis functions and is used to penalize excessive
curvature variations. This regularization promotes smoother spline
contours while reducing abrupt shape changes.

Parameters
----------
num_ctrlpts : int
    Number of control points defining the B-spline curve.

spline_order : int
    Order of the B-spline.

Returns
-------
numpy.ndarray
    Cyclic bending matrix used for spline regularization. The matrix
    has dimensions equal to the number of control points and is
    incorporated into the least-squares optimization to enforce
    contour smoothness.
"""


def ComputeBendingMatrixCyclic(num_ctrlpts, spline_order):
    # Compute the interval length
    s = 1. / (num_ctrlpts - spline_order)

    # Base block for the bending matrix
    Bb = np.array([[2, -3, 0, 1],
                   [-3, 6, -3, 0],
                   [0, -3, 6, -3],
                   [1, 0, -3, 2]]) / (6 * s ** 3)

    # Initialize the bending matrix
    bendingMatrix = np.zeros((num_ctrlpts, num_ctrlpts))

    # Fill the bending matrix for a cycle
    for i in range(num_ctrlpts):
        for di in range(-1, 3):
            for dj in range(-1, 3):
                i_mod = (i + di) % num_ctrlpts
                j_mod = (i + dj) % num_ctrlpts
                bendingMatrix[i_mod, j_mod] += Bb[di + 1, dj + 1]

    return bendingMatrix


# ============================================================
# ComputeControlPoints2
# ============================================================
"""
Compute the optimal B-spline control points using a regularized
least-squares approximation.

This function is the core of the B-spline fitting procedure. It
estimates the control points that best approximate the input contour
while balancing data fidelity and contour smoothness through a
regularization term.

The optimization minimizes a cost function composed of:
    - a data fitting term, ensuring that the spline closely follows
      the input contour points;
    - a smoothness term, controlled by the regularization parameter
      lambda_val, which penalizes excessive curvature variations.

Parameters
----------
points : numpy.ndarray
    Input contour points to be approximated by the B-spline curve.

spline_order : int
    Order of the B-spline.

lambda_val : float
    Regularization parameter controlling the trade-off between
    contour accuracy and smoothness.

num_ctrlpts : int
    Number of control points used to represent the B-spline curve.

Returns
-------
numpy.ndarray
    Estimated control points defining the regularized B-spline
    approximation of the input contour.
"""


def create_closed_knot_vector(num_ctrlpts, spline_order):
    num_segments = num_ctrlpts - 1
    num_internal_knots = num_segments + spline_order + 1
    internal_knots = np.linspace(0, 1, num_internal_knots - 2, endpoint=True)
    knots = np.concatenate(([0] * (spline_order - 1), internal_knots, [1] * (spline_order - 1)))
    return knots


# Create a closed periodic contour by copying the spline_order control points
def create_closed_periodic_knot_vector(num_ctrlpts, spline_order):
    num_ctrlpts_periodic = num_ctrlpts + spline_order
    num_knots = num_ctrlpts_periodic + spline_order + 1
    knots = np.linspace(0, 1, num_knots, endpoint=True)
    return knots


def cumulative_chord_length(points, chord_len_min=0.0, chord_len_max=1.0):
    chord_length_list = [0.0]

    for i in range(len(points) - 1):
        seg = points[i + 1] - points[i]
        length = np.sqrt(np.dot(seg, seg))
        chord_length_list.append(chord_length_list[-1] + length)

    seg = points[i + 1] - points[i]
    length = np.sqrt(np.dot(seg, seg))
    total_chord_length = chord_length_list[-1] + length

    for i in range(len(chord_length_list)):
        chord_length_list[i] = chord_len_min + (chord_len_max - chord_len_min) * chord_length_list[
            i] / total_chord_length

    return chord_length_list


# Approximation aux moindres carrés
# Pour avoir une courbe fermée périodique, on duplique un nombre de points de contrôle égal à l'ordre de la B-spline
def ComputeControlPoints2(points, spline_order, lambda_val, num_ctrlpts):
    num_rows = len(points)

    # knots = create_closed_knot_vector(num_ctrlpts, spline_order)
    knots = create_closed_periodic_knot_vector(num_ctrlpts, spline_order)

    # Calculate the N basis matrix
    num_cols = len(knots) - spline_order - 1
    N_matrix = np.zeros((num_rows, num_cols))

    # t_values = np.linspace(0, 1, num_cols, endpoint=False)	# Il faudrait normalement considérer la longueur des cordes cumulées pour que la densité des noeuds soit homogène
    num_ctrlpts_periodic = num_ctrlpts + 2 * spline_order
    t_values = cumulative_chord_length(points, spline_order / num_ctrlpts_periodic, num_ctrlpts / num_ctrlpts_periodic)

    for i in range(num_rows):
        t = t_values[i]
        for j in range(num_cols):
            N_matrix[i, j] = NonUniformBSplineFunctionRecursive2(knots, spline_order, j, t)
    N_matrix[num_rows - 1, num_cols - 1] = 1.0

    # Compute the cyclic bending matrix
    bendingMatrix = ComputeBendingMatrixCyclic(num_cols, spline_order)

    # Solve for control points
    B = np.linalg.pinv(N_matrix.T @ N_matrix + lambda_val * bendingMatrix) @ N_matrix.T @ points

    return B  # Return both control points and knots as a tuple


def mean_squared_error(true_points, predicted_points, squared=True):
    mse = np.mean(np.square(np.subtract(true_points, predicted_points)))
    return mse if squared else np.sqrt(mse)


def compute_LOO_error(points, spline_order, lambda_vals, num_ctrlpts):
    num_points = len(points)
    errors = np.zeros(len(lambda_vals))

    for idx, lambda_val in enumerate(lambda_vals):
        loo_errors = []
        knots_loo = create_closed_periodic_knot_vector(num_ctrlpts, spline_order)
        # t_values_loo = np.linspace(0, 1, len(knots_loo) - spline_order - 1, endpoint=False)
        num_ctrlpts_periodic = num_ctrlpts + 2 * spline_order
        t_values_loo = cumulative_chord_length(points, spline_order / num_ctrlpts_periodic,
                                               num_ctrlpts / num_ctrlpts_periodic)
        for loo_index in range(num_points):
            # Créez un ensemble de points d'entraînement en excluant le point actuel
            train_points = np.delete(points, loo_index, axis=0)

            # Utilisez votre fonction existante pour calculer les points de contrôle pour l'ensemble d'entraînement
            computed_control_points = ComputeControlPoints2(train_points, spline_order, lambda_val, num_ctrlpts)

            # Préparez-vous à estimer le point exclu
            t_loo = t_values_loo[loo_index]
            N_loo = [NonUniformBSplineFunctionRecursive2(knots_loo, spline_order, i, t_loo) for i in
                     range(len(knots_loo) - spline_order - 1)]

            # Estimer le point exclu en utilisant les points de contrôle calculés
            predicted_point = np.dot(N_loo, computed_control_points)  # S'assure que cette opération est valide

            # Calculez l'erreur pour le point exclu
            # loo_errors.append(mean_squared_error([points[loo_index]], [predicted_point]))
            loo_errors.append(np.linalg.norm(points[loo_index] - predicted_point))
        # Calculez l'erreur moyenne pour cette valeur de lambda
        errors[idx] = np.mean(loo_errors)

    return errors

def evaluate_bspline_from_basis(control_points, spline_order, num_ctrlpts, num_samples=300):
    knots = create_closed_periodic_knot_vector(num_ctrlpts, spline_order)
    num_cols = len(knots) - spline_order - 1

    num_ctrlpts_periodic = num_ctrlpts + 2 * spline_order
    t_min = spline_order / num_ctrlpts_periodic
    t_max = num_ctrlpts / num_ctrlpts_periodic

    t_values = np.linspace(
        t_min,
        t_max,
        num_samples,
        endpoint=False
    )

    N_eval = np.zeros((num_samples, num_cols))

    for i, t in enumerate(t_values):
        for j in range(num_cols):
            N_eval[i, j] = NonUniformBSplineFunctionRecursive2(
                knots,
                spline_order,
                j,
                t
            )

    curve_points = N_eval @ control_points

    # fermeture explicite
    curve_points = np.vstack([curve_points, curve_points[0]])

    return curve_points[:, 0], curve_points[:, 1]


# ============================================================
# Main script
# ============================================================

# TODO: Put here the path to the folder containing the binary masks
MASK_DIR = "path/to/your/masks_endo"

# TODO: Put here the path where the results will be saved
SAVE_DIR = "path/to/save/results"

IMAGE_EXTENSION = "*.png"

NUM_CTRLPTS = 12
SPLINE_ORDER = 3
NUM_BSPLINE_SAMPLES = 300
NUM_NODAL_POINTS = 12

# Regularization values tested using LOOCV
LAMBDA_VALS = np.arange(1e-8, 2.1e-7, 1e-8)

os.makedirs(SAVE_DIR, exist_ok=True)

image_list = glob(os.path.join(MASK_DIR, IMAGE_EXTENSION))

json_data = {}
lambda_sum = 0.0
error_sum = 0.0
n_images = 0

for image_path in image_list:

    # Read the binary mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Warning: image could not be loaded: {image_path}")
        continue

    # Convert the image into a binary mask
    _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Extract external contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"Warning: no contour found for: {image_path}")
        continue

    # Select the largest contour
    largest_contour = max(contours, key=lambda c: cv2.arcLength(c, True))

    contour_points = np.squeeze(largest_contour).astype(np.float32)

    if contour_points.ndim != 2 or contour_points.shape[0] < NUM_CTRLPTS:
        print(f"Warning: invalid contour for: {image_path}")
        continue

    # --------------------------------------------------------
    # Select the optimal lambda using LOOCV
    # --------------------------------------------------------

    loo_errors = compute_LOO_error(
        contour_points,
        SPLINE_ORDER,
        LAMBDA_VALS,
        NUM_CTRLPTS
    )

    optimal_lambda_idx = np.argmin(loo_errors)
    optimal_lambda = LAMBDA_VALS[optimal_lambda_idx]
    optimal_error = loo_errors[optimal_lambda_idx]

    print(
        f"{os.path.basename(image_path)} | "
        f"optimal lambda = {optimal_lambda:.2e} | "
        f"LOO error = {optimal_error:.4f}"
    )

    lambda_sum += optimal_lambda
    error_sum += optimal_error
    n_images += 1

    # --------------------------------------------------------
    # Compute B-spline control points
    # --------------------------------------------------------

    control_points = ComputeControlPoints2(
        contour_points,
        SPLINE_ORDER,
        optimal_lambda,
        NUM_CTRLPTS
    )

    bspline_x, bspline_y = evaluate_bspline_from_basis(
        control_points,
        SPLINE_ORDER,
        NUM_CTRLPTS,
        num_samples=NUM_BSPLINE_SAMPLES
    )

    # --------------------------------------------------------
    # Sample nodal points from the fitted B-spline curve
    # --------------------------------------------------------

    sampled_indices = np.linspace(
        0,
        len(bspline_x) - 2,  # avoid duplicated closing point
        NUM_NODAL_POINTS,
        dtype=int
    )

    sampled_points = [
        [
            [float(bspline_x[i]), float(bspline_y[i])]
            for i in sampled_indices
        ]
    ]

    json_data[os.path.basename(image_path)] = sampled_points

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    image_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contour, [largest_contour], -1, (0, 255, 0), 2)

    bspline_y_display = image.shape[0] - bspline_y

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.flip(image_contour, 0), cmap="gray", origin="lower")
    plt.plot(
        bspline_x,
        bspline_y_display,
        "r-",
        label="Fitted B-spline curve"
    )

    plt.xlim([0, image.shape[1]])
    plt.ylim([0, image.shape[0]])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(f"B-spline fitting | lambda = {optimal_lambda:.2e}")

    figure_name = f"bspline_{os.path.basename(image_path)}"
    figure_path = os.path.join(SAVE_DIR, figure_name)

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Save nodal points
# ============================================================

json_file_path = os.path.join(SAVE_DIR, "nodal_points_bspline.json")

with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(json_data, json_file, indent=4)

if n_images > 0:
    mean_lambda = lambda_sum / n_images
    mean_error = error_sum / n_images

    print("\nProcessing completed.")
    print(f"Number of valid images: {n_images}")
    print(f"Mean optimal lambda: {mean_lambda:.2e}")
    print(f"Mean LOO error: {mean_error:.4f}")
    print(f"Nodal points saved in: {json_file_path}")
else:
    print("No valid image was processed.")