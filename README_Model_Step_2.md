BS2GDL: B-Spline Geometrically-Guided CNN Models for LV Segmentation


Overview : 

BS2GDL is a deep learning framework for segmenting the left ventricle (LV) myocardium in medical imaging. The architecture is designed in two stages:

    CNN_Step_1: Initial nodal point regression.
    CNN_Step_2: Segmentation refinement using geometric constraints.


This document focuses on Step 1.

CNN_Step_1: Initial Nodal Point Regression

Description :

The first stage of BS2GDL, CNN_Step_1, regresses key nodal points representing the LV contour. These nodal points are parameters for a parametric B-spline model, enabling a structured representation of the LV boundaries.
Features

    Encoder-Decoder Architecture: Captures hierarchical spatial features and predicts nodal points.
    Loss Function: Uses Root Mean Squared Error (RMSE) to optimize the regression of nodal points.


Installation : 
Prerequisites : 

    Install required Python packages:

    pip install tensorflow numpy scikit-learn matplotlib

Usage : 
Data Preparation : 

Organize your dataset as follows:

    Images: <path_to_images>
    Masks: <path_to_masks>
    Nodal Points: JSON file specifying the nodal points for each image.

Training : 

Run the following command to train the model :

python train_step1.py --image_folder <path_to_images> --mask_epi_folder <path_to_masks> \
                      --json_file <path_to_nodal_points> --output_dir <path_to_save_outputs>


Outputs : 

    Predicted nodal points for LV contour approximation.
    Trained model weights stored in <output_dir>.
	
	

Project Files : 


data_loader.py

Purpose: Handles data loading and preprocessing for training and evaluation.

    Key Functions:
        load_data: Loads images, masks, contours, nodal points, and segmentation data from the specified directories.
        load_json_data: Reads and processes JSON files containing nodal point annotations.
    Usage:
        Organizes and normalizes data for compatibility with the training pipeline.
        Supports generation of background masks and concatenation of relevant segmentation data.



bspline.py

Purpose: Implements the B-spline-based geometric modeling.

    Key Components:
        BSplineDetector: A class to compute control points and evaluate B-spline curves for nodal point regression.
            Includes methods to generate circle points, compute control points, and evaluate B-spline curves.
        BSplineLayer: A TensorFlow custom layer to generate B-spline curves dynamically during model inference.
            Generates smooth curves based on predicted nodal points.
    Usage:
        Integrated within the CNN to ensure geometric consistency of the LV boundaries.
        Produces smooth and closed contours for accurate representation.



BS2GDL_Model_Step_1.py

Purpose: Defines the Step 1 model architecture for nodal point regression.

    Key Features:
        AttentionGate: A custom layer to enhance feature map importance during decoding.
        CustomUNetModel: A U-Net-based architecture with integrated attention mechanisms and support for B-spline curve generation.
            Includes both encoding and decoding paths for hierarchical feature extraction and nodal point regression.
    Usage:
        Outputs predicted nodal points for the LV contour.
        Provides intermediate layers for visualization and debugging if needed.


train_step_1.py

Purpose: The main script to train and validate CNN_Step_1.

    Key Features:
        Data Loading: Loads the prepared dataset using data_loader.py.
        Model Compilation: Configures the CustomUNetModel with a loss function (RMSE_loss) and optimizer.
        Training Pipeline:
            Divides the dataset into training and validation sets.
            Implements callbacks like EarlyStopping and ReduceLROnPlateau for training stability.
        Outputs:
            Trained model weights (model_weights.h5).
            Training logs for loss and accuracy.
    Usage:
        Run the script with specified arguments to start the training process.