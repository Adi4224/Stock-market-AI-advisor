"""
Verification script for stock price prediction pipeline.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.middle_end.prediction_pipeline import predict_stock_price, predict_stock_movement

def test_predictions():
    print("=" * 60)
    print("TESTING STOCK PREDICTIONS FOR AAPL")
    print("=" * 60)
    
    for model in ["random_forest", "xgboost", "svm"]:
        print(f"\nModel: {model.upper()}")
        
        # Test price prediction
        price_res = predict_stock_price("AAPL", model)
        if "error" in price_res:
            print(f"  [FAIL] Price Prediction Failed: {price_res['error']}")
        else:
            print(f"  [SUCCESS] Price Prediction Success:")
            print(f"     Current Price: {price_res.get('current_price')}")
            print(f"     Predicted Price: {price_res.get('predicted_price')}")
            print(f"     Change %: {price_res.get('change_pct')}%")
            print(f"     Model Used: {price_res.get('model_used')}")
            print(f"     Is Demo: {price_res.get('is_demo')}")
            
        # Test movement prediction
        move_res = predict_stock_movement("AAPL", model)
        if "error" in move_res:
            print(f"  [FAIL] Movement Prediction Failed: {move_res['error']}")
        else:
            print(f"  [SUCCESS] Movement Prediction Success:")
            print(f"     Current Price: {move_res.get('current_price')}")
            print(f"     Predicted Movement: {move_res.get('predicted_movement')}")
            print(f"     Probability: {move_res.get('probability')}")
            print(f"     Is Demo: {move_res.get('is_demo')}")

if __name__ == "__main__":
    test_predictions()
