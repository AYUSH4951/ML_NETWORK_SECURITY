import os
import unittest
import pandas as pd
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

class TestMLPipeline(unittest.TestCase):
    def test_model_artifacts_exist(self):
        self.assertTrue(os.path.exists("final_model/model.pkl"), "final_model/model.pkl not found")
        self.assertTrue(os.path.exists("final_model/preprocessor.pkl"), "final_model/preprocessor.pkl not found")

    def test_model_loading(self):
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(model)
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        self.assertIsNotNone(network_model)

    def test_prediction_inference(self):
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        
        test_data_path = "valid_data/test.csv"
        if os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            predictions = network_model.predict(df)
            self.assertEqual(len(predictions), len(df))
            self.assertTrue(set(predictions).issubset({0, 1, 0.0, 1.0}))

    def test_app_routes(self):
        from app import app
        route_paths = [route.path for route in app.routes]
        self.assertIn("/", route_paths)
        self.assertIn("/docs", route_paths)
        self.assertIn("/train", route_paths)
        self.assertIn("/predict", route_paths)

if __name__ == "__main__":
    unittest.main()
