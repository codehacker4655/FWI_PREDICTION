import pickle
from flask import Flask, request, render_template
import pandas as pd  # <-- add pandas

application = Flask(__name__)
app = application

# Load model & scaler
ridge_model = pickle.load(open('models/model.pkl','rb'))
scale = pickle.load(open('models/sc.pkl','rb'))

# Must match the columns used during training in the SAME order
FEATURE_COLS = ["Temperature","RH","Ws","Rain","FFMC","DMC","ISI","Classes","Region"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        # Read form values (convert to float/int as appropriate)
        vals = {
            "Temperature": float(request.form.get("Temperature")),
            "RH":          float(request.form.get("RH")),
            "Ws":          float(request.form.get("Ws")),
            "Rain":        float(request.form.get("Rain")),
            "FFMC":        float(request.form.get("FFMC")),
            "DMC":         float(request.form.get("DMC")),
            "ISI":         float(request.form.get("ISI")),
            "Classes":     float(request.form.get("Classes")),  # or int(...)
            "Region":      float(request.form.get("Region")),   # or int(...)
        }

        # Build a 1-row DataFrame with the expected columns & order
        X_in = pd.DataFrame([[vals[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

        # Scale with the trained scaler (now no warning)
        X_scaled = scale.transform(X_in)

        # Predict
        y_pred = ridge_model.predict(X_scaled)

        # Convert to a scalar for Jinja
        result_value = float(y_pred[0])
        # Optional: round it
        result_value = round(result_value, 3)

        return render_template("home.html", result=result_value)

    # GET
    return render_template("home.html")

if __name__ == "__main__":
    print("Starting Flask…")
    app.run(debug=True, host="127.0.0.1", port=5000)
