from flask import Blueprint, current_app
from api.utils.helpers import get_db_connection, success_response, error_response

segments_bp = Blueprint('segments', __name__)

@segments_bp.route('/segments', methods=['GET'])
def get_all_segments():
    """
    Returns the segment classification for all customers.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id, segment, annual_income, spending_score, purchase_frequency FROM customers")
        rows = cursor.fetchall()
        conn.close()
        
        # Format rows as dictionary
        customer_segments = [
            {
                "customer_id": row["customer_id"],
                "segment": row["segment"],
                "annual_income": row["annual_income"],
                "spending_score": row["spending_score"],
                "purchase_frequency": row["purchase_frequency"]
            }
            for row in rows
        ]
        
        return success_response(customer_segments)
    except Exception as e:
        current_app.logger.error(f"Error fetching all segments: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)

@segments_bp.route('/segments/<string:customer_id>', methods=['GET'])
def get_customer_segment(customer_id):
    """
    Returns the segment for a specific customer.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT customer_id, segment, annual_income, spending_score, purchase_frequency FROM customers WHERE customer_id = ?",
            (customer_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return error_response(f"Customer {customer_id} not found", 404)
            
        segment_data = {
            "customer_id": row["customer_id"],
            "segment": row["segment"],
            "annual_income": row["annual_income"],
            "spending_score": row["spending_score"],
            "purchase_frequency": row["purchase_frequency"]
        }
        
        return success_response(segment_data)
    except Exception as e:
        current_app.logger.error(f"Error fetching segment for customer {customer_id}: {str(e)}")
        return error_response(f"Internal Server Error: {str(e)}", 500)
