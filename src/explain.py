import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "./data/unsw_nb15_synthetic.csv"
MODEL_PATH = "./reports/rf_model.pkl"

def explain_model():
    print("📥 Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop("label", axis=1)

    print("🔍 Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("📊 Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("./reports/shap_summary_plot.png")
    print("✅ Saved: reports/shap_summary_plot.png")

if __name__ == "__main__":
    explain_model()
