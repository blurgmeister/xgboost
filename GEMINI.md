# Project Mandates: Leaf Node Prediction Interpolation in XGBoost

This repository is a fork of the public XGBoost project, specifically focused on the development and implementation of leaf node prediction interpolation.

## User Statement and Goal

"The purpose of this repository is to facilitate development of leaf node prediction interpolation. The aim is to allow the user to select which features they want interpolation to be applied to and by how much. There will be two interpolation methods, linear and sigmoid. We will interpolate symmetrically either side of the parent node split point, with the leaf node predictions serving as the values to interpolate between. The user will select the 'span' (ie the distance either side of the spliit point) to interpolate over. I am interested in this being applied during model training and at inference, though it may be easier to start with the inference application to workout the necessary logic before embedding in the training steps. The interpolation is to apply only to terminal leaf nodes that share the same immediate parent node, not between cousin or more distant nodes. So we will only interpolate between pairs of sibling leaf nodes at the same depth in a tree."

## Implementation Requirements

1.  **Interpolation Target:**
    -   Applies *only* to terminal sibling leaf nodes (those sharing the same immediate parent).
    -   Does *not* apply between cousin nodes or more distant relatives.
    -   Interpolation occurs symmetrically on both sides of the parent node's split point.

2.  **User Configuration:**
    -   **Feature Selection:** Users must be able to specify which features should trigger interpolation.
    -   **Interpolation Span:** Users must be able to define the "span" (distance) on either side of the split point over which interpolation occurs. The span is intended to apply multiplicatively like a percentage of the split point. eg interpolation_distance = span * abs(split). Where span >= 0. 
    -   **Interpolation Method:** Support for both `linear` and `sigmoid` methods. If span = 0 for a selected feature, then no interpolation will be applied.

3.  **Development Phases:**
    -   **Phase 1: Baseline examples.** Create 2 python scripts and download 2 example datasets. One dataset for a logistic regression or classification task where the desired output is a probability score, and the other example dataset a more standard regression problem. Save the python scripts to the directory /workspaces/xgboost/tests/test_interpolation/ . Use the first script to download the necessary example datasets and save them to /workspaces/test_output/regression/ for the regression example and to directory /workspaces/test_output/classification/ for the classification/logistic-regression example. The second pyhton script (also to be saved in the /workspaces/xgboost/tests/test_interpolation/ directory) will create and run 3 separate xgboost models each of depth 1,2 and 3 using the original xgboost code for each dataset (regression/classification). These will be used as a baseline for comparison against the version of code that will include interpolation. This second python script will include code to generate partial dependency plots (PDP) and individual conditional expectation (ICE) plots. In the ICE plots, rescale all the individual traces in the plot (ie divide) by the value of each trace at the mode or median of the input value distribution. Save a plot for each of the top 10 features in each model as png files and as html files (using the plotly package for html). When saving the plotly ICE plots as html, use only a subset of the traces to avoid overcrowding the plot and to keep the html file size manageable. Create the pyhton script in the /workspaces/xgboost/tests/test_interpolation/ directory and save the outputs of the regression model PDP and ICE charts to the directory /workspaces/test_output/regression/original_code/ . For the classification model, save the output PDP and ICE charts to the directory /workspaces/test_output/classification/original_code/ . Name the model output charts appropriately to reflect the model's depth and the feature shown in the chart. Set a random seed for each model run for reproducibility. 
    
    -   Phase 2: Inference Application. Implement the interpolation logic within the prediction engine (`src/predictor/`) to validate the core logic and user parameters.
        -   **Configuration Parameters:**
            -   `interpolation_features`: A vector of feature indices to apply interpolation to.
            -   `interpolation_spans`: A vector of floats representing the span for each feature.
            -   `interpolation_method`: Either `linear` or `sigmoid`.
            -   `interpolation_min_span`: A floor (default 0.0001) to avoid division by zero.
        -   **Integration:** These parameters will be integrated into `GBTreeModelParam` and will be accessible via the Python API. The core logic will be implemented in `src/predictor/predict_fn.h` to be shared between CPU and GPU predictors.
        -   **Terminal Sibling Constraint:** Interpolation will only occur when a node's children are both leaves.


    -   **Phase 3: Training Integration.** Embed the interpolation logic into the tree-building and training process.

    -   **Phase 4: Comparison examples.** We rerun the same models from phase 1 using the new code which includes interpolation logic so charts can be compared. Create a new python script for this which saves the PDP and ICE charts to the directory /workspaces/test_output/regression/new_code for the regression model. And for the classification model, it saves the output PDP and ICE charts to the directory /workspaces/test_output/classification/new_code .

4.  **Mathematical Logic:**
    -   Use the leaf node predictions as the target values for the interpolation.
    -   The interpolation should provide a smooth transition between the predictions of sibling leaves based on the input feature value's proximity to the split point.
    -   Because the span is multiplicative, we will give the user an additional minimum_span argument which will default to 0.0001 that will apply as a floor on the interpolation_distance = max(minimum_span, span * abs(split)). This will avoid divide by zero errors in the calculations when a split point is exactly 0.
    -   Below are 2 examples of psudo code for each of the linear and sigmoid formulas when interpolating.
    -   Both versions of each use the same underlying logic, but one version packages it within conditional logic/filtering of the feature values whilst the other applies it directly.

    -   Linear (option 1 - conditional version): 

        IF feature_value < split - MAX(minimum_span, span*ABS(split)) THEN 
            RETURN left_leaf_prediction 
        ELSE IF feature_value > split + MAX(minimum_span, span*ABS(split)) THEN 
            RETURN right_leaf_prediction
        ELSE
            RETURN (right_leaf_prediction - left_leaf_prediction)/(2*MAX(minimum_span, span*ABS(split)))*(feature_value - split) + ( right_leaf_prediction + left_leaf_prediction)/2

    -   Linear (option 2 - direct version):

        MIN(MAX(left_leaf_prediction, right_leaf_prediction),MAX(MIN(left_leaf_prediction, right_leaf_prediction), (right_leaf_prediction - left_leaf_prediction)/(2*MAX(minimum_span, span*ABS(split)))*(feature_value - split) + ( right_leaf_prediction + left_leaf_prediction)/2))

    -   Sigmoid (option 1 - conditional version):

        IF feature_value < split - MAX(minimum_span, span*ABS(split)) THEN 
            RETURN left_leaf_prediction 
        ELSE IF feature_value > split + MAX(minimum_span, span*ABS(split)) THEN 
            RETURN right_leaf_prediction
        ELSE
            RETURN left_leaf_prediction + (right_leaf_prediction - left_leaf_prediction) * (1-(EXP(5/(MAX(minimum_span, span*ABS(split)))*(feature_value - split))+1)^(-1))

    -   Sigmoid (option 2 - direct version):

        left_leaf_prediction + (right_leaf_prediction - left_leaf_prediction) * (1-(EXP(5/(MAX(minimum_span, span*ABS(split)))*(feature_value - split))+1)^(-1))


## Engineering Standards

-   **Adherence to XGBoost Conventions:** All code must follow the established coding standards and architectural patterns of the XGBoost project (C++, Python, R, and JVM).
-   **Configuration Handling:** Integrate interpolation parameters into the existing XGBoost parameter system (e.g., using `DMLC_DECLARE_PARAMETER` and JSON serialization).
-   **Testing:** Comprehensive unit and integration tests must be provided for both interpolation methods, ensuring correctness at the inference level first.
-   **Documentation:** Clear documentation of the new parameters and their effects on model behavior.

## Phase 2 Implementation Plan:

1. Configuration & Parameters
  The new parameters must be added to the GBTreeModelParam struct in src/gbm/gbtree_model.h. This will ensure they are
  part of the model's configuration and are correctly serialized during model save/load.


   * File: src/gbm/gbtree_model.h
   * Changes:
       * Define an enum class InterpolationMethod { kLinear, kSigmoid }.
       * Add fields to GBTreeModelParam:
           * interpolation_features: Using common::ParamArray<float> to store feature indices.
           * interpolation_spans: Using common::ParamArray<float> to store the span for each feature.
           * interpolation_method: Of type InterpolationMethod.
           * interpolation_min_span: Float with a default of 0.0001f.
       * Register these fields in DMLC_DECLARE_PARAMETER(GBTreeModelParam).


  2. Prediction View Integration
  To make these parameters available during the prediction process (especially for GPU), they need to be passed through
  GBTreeModelView.


   * File: src/predictor/gbtree_view.h
   * Changes:
       * Update the GBTreeModelView class to store interpolation settings.
       * To optimize lookup during prediction, the constructor can pre-process interpolation_features and
         interpolation_spans into a std::vector<float> (or a device-compatible array for GPU) of size num_features, where
         a non-zero value indicates the span for that feature.

  3. Core Interpolation Logic
  The mathematical logic will be implemented as a shared function in the prediction header.


   * File: src/predictor/predict_fn.h
   * Changes:
       * Add a function Interpolate(float fvalue, float split, float left_val, float right_val, float span, float
         min_span, InterpolationMethod method).
       * Implement the Linear and Sigmoid formulas provided in GEMINI.md.


  4. CPU Predictor Implementation
  The CPU predictor needs to intercept the tree traversal to apply interpolation.


   * File: src/predictor/cpu_predictor.cc
   * Changes:
       * Modify scalar::PredValueByOneTree (and its multi-target counterpart).
       * Instead of calling GetLeafIndex which only returns the final leaf, the traversal loop should check if the
         current node's children are both terminal leaves.
       * If they are, and if the split feature is configured for interpolation, call the Interpolate function.


  5. GPU Predictor Implementation
  Similarly, the GPU kernel needs to be updated.


   * File: src/predictor/gpu_predictor.cu
   * Changes:
       * Update the GetLeafIndex (or create a GetLeafValue) device function.
       * Ensure the interpolation parameters are available in the device memory (passed via the GBTreeModelView or kernel
         arguments).


  6. Python API Exposure
  Since the parameters are registered via DMLC_DECLARE_PARAMETER, they will automatically be accessible via the Booster's
  set_param and when passing a params dictionary in Python. However, we should ensure the Python wrapper correctly
  handles passing lists/arrays as strings if needed (though ParamArray usually handles this).


  7. Summary of Implementation Points

   Component : File Path : Action 
   * Model Params : src/gbm/gbtree_model.h : Add interpolation fields to GBTreeModelParam. 
   * Shared Logic : src/predictor/predict_fn.h : Add Interpolate math function.                
   * Predictor View : src/predictor/gbtree_view.h : Pass params to CPU/GPU predictors.            
   * CPU Engine : src/predictor/cpu_predictor.cc : Modify traversal loop to apply interpolation. 
   * GPU Engine : src/predictor/gpu_predictor.cu : Update CUDA kernel for interpolation.         
  
