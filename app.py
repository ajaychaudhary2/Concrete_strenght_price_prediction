from flask import Flask, render_template, request
import pandas as pd
from Concrete_Strenght_prediction.pipelines.prediction_pipeline import Prediction_pipeline, Custom_data

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # Render the form to input data
        return render_template("index.html")
    else:
        # Collect data from the form
        data = {
            "cement": float(request.form.get("cement")),
            "blast_furnace_slag": float(request.form.get("blast_furnace_slag")),
            "fly_ash": float(request.form.get("fly_ash")),
            "water": float(request.form.get("water")),
            "superplasticizer": float(request.form.get("superplasticizer")),
            "coarse_aggregate": float(request.form.get("coarse_aggregate")),
            "fine_aggregate": float(request.form.get("fine_aggregate")),
            "age": int(request.form.get("age"))
        }

        # Create an instance of the Custom_data class
        custom_data = Custom_data(**data)

        # Convert the input data into a DataFrame
        input_df = custom_data.get_data_as_df()

        # Create an instance of the Prediction_pipeline
        predict_pipeline = Prediction_pipeline()

        # Make predictions
        try:
            prediction = predict_pipeline.make_prediction(input_df)
            result = round(prediction[0], 2)  # Round to 2 decimal places
            return render_template("result.html", prediction_result=result)
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
