import unittest
import json
from api.app import create_app

class TestFlaskAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Creates a Flask test client for API routing validation.
        """
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_health_check(self):
        """
        Tests GET /api/health endpoint.
        """
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['status'], 'healthy')
        self.assertIn('models_loaded', data['data'])

    def test_get_all_segments(self):
        """
        Tests GET /api/segments endpoint.
        """
        response = self.client.get('/api/segments')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertTrue(isinstance(data['data'], list))
        self.assertGreater(len(data['data']), 0)
        self.assertIn('customer_id', data['data'][0])
        self.assertIn('segment', data['data'][0])

    def test_get_customer_segment(self):
        """
        Tests GET /api/segments/<customer_id> endpoint.
        """
        # Test valid customer
        response = self.client.get('/api/segments/C001')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['customer_id'], 'C001')
        
        # Test invalid customer
        response = self.client.get('/api/segments/C9999')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'error')

    def test_predict_purchase(self):
        """
        Tests POST /api/predict/purchase with valid and invalid inputs.
        """
        valid_payload = {
            'age': 30,
            'annual_income': 60000,
            'spending_score': 70,
            'loyalty_years': 4,
            'average_order_value': 150.5
        }
        
        # Test valid post request
        response = self.client.post(
            '/api/predict/purchase',
            data=json.dumps(valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertIn('will_purchase', data['data'])
        self.assertIn('purchase_probability', data['data'])
        
        # Test invalid payload (missing loyalty_years)
        invalid_payload = {
            'age': 30,
            'annual_income': 60000,
            'spending_score': 70,
            'average_order_value': 150.5
        }
        response = self.client.post(
            '/api/predict/purchase',
            data=json.dumps(invalid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'error')

    def test_get_top_churn_risk(self):
        """
        Tests GET /api/churn-risk endpoint.
        """
        response = self.client.get('/api/churn-risk')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']), 20)
        self.assertIn('churn_probability', data['data'][0])

    def test_get_customer_churn_risk(self):
        """
        Tests GET /api/churn-risk/<customer_id> endpoint.
        """
        response = self.client.get('/api/churn-risk/C001')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['customer_id'], 'C001')
        self.assertIn('churn_probability', data['data'])
        
        # Test invalid customer
        response = self.client.get('/api/churn-risk/C9999')
        self.assertEqual(response.status_code, 404)

    def test_get_recommendations(self):
        """
        Tests GET /api/recommend/<customer_id> endpoint.
        """
        response = self.client.get('/api/recommend/C001')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['customer_id'], 'C001')
        self.assertEqual(len(data['data']['recommendations']), 5)

    def test_get_dashboard_stats(self):
        """
        Tests GET /api/dashboard/stats endpoint.
        """
        response = self.client.get('/api/dashboard/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        
        # Verify keys
        stats = data['data']
        self.assertIn('total_customers', stats)
        self.assertIn('churn_risk_percent', stats)
        self.assertIn('avg_spending_score', stats)
        self.assertIn('top_segment', stats)
        self.assertIn('segment_breakdown', stats)
        self.assertIn('aov_by_segment', stats)
        self.assertIn('monthly_revenue', stats)

if __name__ == "__main__":
    unittest.main()
