# ml-assignment-2-2025AB05027

# Problem Statement:
    # Predict/ classify the loan approval status basis of the 12 parameters provided in the data set. 
    # This is a supervised classification problem, where the model learns from historical loan data and predicts whether a new loan application will be Approved or Rejected

# Dataset Description:
# The dataset contains loan application records with 12 input features (independent variables)
    # and 1 target column representing the loan approval decision.
    #
    # Target Column:
    # - loan_status (Approved / Rejected)
    #
    # Feature Columns (12):
    # - no_of_dependents -Numeric
    # - education - Categorical
    # - self_employed - Categorical
    # - income_annum - Numerical
    # - loan_amount - Numerical
    # - loan_term - Numerical
    # - cibil_score - Numerical
    # - residential_assets_value - Numerical
    # - commercial_assets_value - Numerical
    # - luxury_assets_value - Numerical
    # - bank_asset_value - Numerical
    # - other_loans - Numerical
    #
    # The dataset includes both numerical and categorical variables.
    # Categorical features are encoded using OneHotEncoding and numerical features are scaled using StandardScaler.

# Models used:
    
    | ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1        | MCC      |
    |--------------------------|----------|----------|-----------|----------|-----------|----------|
    | Logistic Regression      | 0.936842 | 0.979008 | 0.938899  | 0.939698 | 0.9372903 | 0.869388 |
    | Decision Tree            | 0.981579 | 0.981162 | 0.981617  | 0.981162 | 0.981591  | 0.961085 |
    | KNN                      | 0.925000 | 0.980891 | 0.924831  | 0.919017 | 0.924874  | 0.840857 |
    | Naive Bayes              | 0.951316 | 0.979242 | 0.951556  | 0.950121 | 0.951393  | 0.897402 |
    | Random Forest (Ensemble) | 0.980263 | 0.998817 | 0.980434  | 0.976183 | 0.980202  | 0.958293 |
    | XGBoost (Ensemble)       | 0.986842 | 0.998960 | 0.986839  | 0.985426 | 0.986833  | 0.972136 |


# Observations on the model behavior and associated implementation on the data set:
    
    | ML Model Name            | observation about model performance | 
    |--------------------------|----------|
    | Logistic Regression      | Overall good at predicting teh correct labels, Strong ability to separate Approved and rejected and mostly stable. However, overall classification quality is weaker than the 3 ensemble models|
    | Decision Tree            | Excellent correctness, very few false positives and a balanced performance. Overall extremely relaible model, learns rules clearly from dataset | 
    | KNN                      | Prediction correctness is lower compared to all other models, Good at separation but still lack accurate predictions overall as it depends on the distance and scaling|
    | Naive Bayes              | Better than KNN & Logistic and balanced / consistent in prediction | 
    | Random Forest (Ensemble) | Excellent overall prediction, very few false negatives and closer to XGBoost model in performance| 
    | XGBoost (Ensemble)       | Best model across all teh 6 models implemented. Strong across the board as it can handle complex patterns & generalizes well| 

# NOTE: 
    This app loads pre-trained models from models/saved/ folder. Run python models/train_models.py if you want to retrain.”

# Instructions to test and see the metrics
    1. Download Test data and upload the same to get teh predictions
    2. Download True label data and uplaod the same to get the detailed metrics around Evaluation & Confusion matrix etc