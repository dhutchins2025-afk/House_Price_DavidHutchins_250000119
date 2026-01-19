from flask import Flask, render_template, request
import joblib, json
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/house_price_model.joblib")
scaler = joblib.load("model/scaler.joblib")
columns = json.load(open("model/columns.json"))

num_cols = [
    "OverallQual","GrLivArea","TotalBsmtSF",
    "GarageCars","BedroomAbvGr","FullBath","YearBuilt"
]

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = {
            "OverallQual": float(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": float(request.form["GarageCars"]),
            "BedroomAbvGr": float(request.form["BedroomAbvGr"]),
            "FullBath": float(request.form["FullBath"]),
            "YearBuilt": float(request.form["YearBuilt"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]

        df[num_cols] = scaler.transform(df[num_cols])
        prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
