from flask import Flask, render_template, request
import sqlite3 as sql
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os


app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
def __predict(path_to_model, features: np.ndarray) -> np.ndarray:
    model = rt.InferenceSession(
        os.path.join(PROJECT_ROOT, path_to_model)
    )
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    return model.run(
        [label_name],
        {input_name: features.astype(np.float32)},
    )


@app.route("/")
def home():
    return render_template("student.html")


@app.route("/enternew")
def new_student():
    return render_template("student.html")


@app.route("/addrec", methods=["POST", "GET"])
def addrec():
    if request.method == "POST":
        try:
            fixed_acidity = request.form["fixed_acidity"]
            volatile_acidity = request.form["volatile_acidity"]
            citric_acid = request.form["citric_acid"]
            residual_sugar = request.form["residual_sugar"]
            chlorides = request.form["chlorides"]
            free_sulfur_dioxide = request.form["free_sulfur_dioxide"]
            total_sulfur_dioxide = request.form["total_sulfur_dioxide"]
            density = request.form["density"]
            ph = request.form["ph"]
            sulphates = request.form["sulphates"]
            alcohol = request.form["alcohol"]

            X = np.array(
                [
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    ph,
                    sulphates,
                    alcohol,
                ]
            )
            X_scaled = __predict("static/scaler_init.onnx", X)
            quality = __predict("static/rf_model_init.onnx", X_scaled)

            with sql.connect("database.db") as con:
                cur = con.cursor()
                cur.execute(
                    """INSERT INTO students (fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    ph,
                    sulphates,
                    alcohol) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        fixed_acidity,
                        volatile_acidity,
                        citric_acid,
                        residual_sugar,
                        chlorides,
                        free_sulfur_dioxide,
                        total_sulfur_dioxide,
                        density,
                        ph,
                        sulphates,
                        alcohol,
                    ),
                )

                con.commit()

                res = "хорошее" if quality == 1 else "плохое"
                msg = f"Вино {res}"
        except:
            con.rollback()
            msg = "Ошибка при загрузке данных"

        finally:
            return render_template("result.html", msg=msg)
            con.close()


@app.route("/list")
def list():
    con = sql.connect("database.db")
    con.row_factory = sql.Row

    cur = con.cursor()
    cur.execute("select * from students")

    rows = cur.fetchall()
    return render_template("list.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
