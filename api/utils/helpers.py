import sqlite3
from flask import jsonify
from config import DATABASE_PATH

def get_db_connection():
    """
    Returns a connection to the SQLite database with Row factory enabled
    for dictionary-like access to database columns.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def success_response(data, status_code=200):
    """
    Returns a formatted success JSON response.
    """
    return jsonify({
        "status": "success",
        "data": data
    }), status_code

def error_response(message, status_code=400):
    """
    Returns a formatted error JSON response.
    """
    return jsonify({
        "status": "error",
        "message": message
    }), status_code

def validate_purchase_input(data):
    """
    Validates that the incoming request payload for purchase prediction
    contains the correct fields, types, and values.
    """
    required_fields = {
        'age': (int, float),
        'annual_income': (int, float),
        'spending_score': (int, float),
        'loyalty_years': (int, float),
        'average_order_value': (int, float)
    }
    
    # Check for missing fields
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
        
    # Check type validity and bounds
    for field, expected_types in required_fields.items():
        val = data[field]
        if not isinstance(val, expected_types):
            return False, f"Field '{field}' must be a numeric value."
            
        # Check logical bounds
        if field == 'age' and not (18 <= val <= 100):
            return False, "Age must be between 18 and 100."
        if field == 'annual_income' and val < 0:
            return False, "Annual income cannot be negative."
        if field == 'spending_score' and not (1 <= val <= 100):
            return False, "Spending score must be between 1 and 100."
        if field == 'loyalty_years' and not (0 <= val <= 50):
            return False, "Loyalty years must be between 0 and 50."
        if field == 'average_order_value' and val < 0:
            return False, "Average order value cannot be negative."
            
    return True, None
