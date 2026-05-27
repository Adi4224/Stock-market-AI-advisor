"""
Verification script for stock price prediction pipeline with multi-horizon support.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.middle_end.prediction_pipeline import predict_stock_price, predict_stock_movement

def test_predictions():
    print("=" * 60)
    print("TESTING STOCK PREDICTIONS FOR AAPL (MULTI-HORIZON)")
    print("=" * 60)
    
    for horizon in ["1 Day", "1 Week", "1 Month"]:
        print(f"\n>>> HORIZON: {horizon} <<<")
        for model in ["random_forest", "xgboost"]:
            print(f"\nModel: {model.upper()}")
            
            # Test price prediction
            price_res = predict_stock_price("AAPL", model, horizon)
            if "error" in price_res:
                print(f"  [FAIL] Price Prediction Failed: {price_res['error']}")
            else:
                print(f"  [SUCCESS] Price Prediction Success:")
                print(f"     Target Date: {price_res.get('prediction_date')}")
                print(f"     Current Price: {price_res.get('current_price')}")
                print(f"     Predicted Price: {price_res.get('predicted_price')}")
                print(f"     Change %: {price_res.get('change_pct')}%")
                print(f"     Confidence: {price_res.get('confidence')}")
                
            # Test movement prediction
            move_res = predict_stock_movement("AAPL", model, horizon)
            if "error" in move_res:
                print(f"  [FAIL] Movement Prediction Failed: {move_res['error']}")
            else:
                print(f"  [SUCCESS] Movement Prediction Success:")
                print(f"     Predicted Movement: {move_res.get('predicted_movement')}")
                print(f"     Probability Certainty: {move_res.get('probability')}")
                print(f"     Confidence Label: {move_res.get('confidence')}")

if __name__ == "__main__":
    test_predictions()
