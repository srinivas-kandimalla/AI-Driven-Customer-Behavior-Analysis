import pandas as pd
from flask import Blueprint, request, current_app
from api.utils.helpers import validate_purchase_input, success_response, error_response

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict/purchase', methods=['POST'])
def predict_purchase_likelihood():
    """
    Predicts if a customer will make a purchase based on demographic and behavior.
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("Invalid or missing JSON payload.", 400)
            
        # Validate input schema
        is_valid, err_msg = validate_purchase_input(data)
        if not is_valid:
            return error_response(err_msg, 400)
            
        # Retrieve models from global app config
        purchase_bundle = current_app.config.get('PURCHASE_MODEL')
        if not purchase_bundle:
            return error_response("Purchase prediction model is not loaded.", 500)
            
        rf_model = purchase_bundle['model']
        scaler = purchase_bundle['scaler']
        feature_cols = purchase_bundle['features']
        
        # Prepare input data matching feature columns exactly
        # Map input keys if they differ (e.g. annual_income vs income)
        input_data = pd.DataFrame([{
            'age': float(data['age']),
            'annual_income': float(data['annual_income']),
            'spending_score': float(data['spending_score']),
            'loyalty_years': float(data['loyalty_years']),
            'average_order_value': float(data['average_order_value'])
        }])
        
        # Scale features using the fitted scaler
        X_scaled = scaler.transform(input_data[feature_cols])
        
        # Predict
        will_purchase = int(rf_model.predict(X_scaled)[0])
        probabilities = rf_model.predict_proba(X_scaled)[0]
        purchase_prob = float(probabilities[1])
        
        result = {
            "will_purchase": will_purchase,
            "purchase_probability": round(purchase_prob, 4),
            "confidence_label": "High Likelihood" if purchase_prob >= 0.7 else ("Moderate Likelihood" if purchase_prob >= 0.4 else "Low Likelihood")
        }
        
        return success_response(result)
    except Exception as e:
        current_app.logger.error(f"Error predicting purchase likelihood: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)
