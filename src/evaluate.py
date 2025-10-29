import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib

DATA_PATH = "./data/unsw_nb15_synthetic.csv"
MODEL_PATH = "./reports/rf_model.pkl"

def evaluate_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    print("📥 Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("🔍 Predicting...")
    y_pred = model.predict(X)

    print("\n✅ Classification Report:")
    print(classification_report(y, y_pred))

    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("./reports/confusion_matrix.png")
    print("📊 Saved: reports/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
