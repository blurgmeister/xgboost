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
    -   **Phase 1: Inference Application.** Implement the interpolation logic within the prediction engine (`src/predictor/`) to validate the core logic and user parameters.
    -   **Phase 2: Training Integration.** Embed the interpolation logic into the tree-building and training process.

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

        IF feature_value < split - MAX(minimum_span, span*ABS(split) THEN 
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
