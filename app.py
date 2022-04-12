import requests
import pickle
import numpy as np

import pandas as pd
from flask import Flask, render_template, url_for,send_file
from flask import request as req

app = Flask(__name__)
output=""

@app.route('/')
def home():
    return render_template("main.html")



@app.route("/textsum", methods=["GET", "POST"])
def Index():
    return render_template("index.html")


@app.route("/Summarize", methods=["GET", "POST"])
def Summarize():
    if req.method == "POST":
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer api_cDqsshiYYdsPmHybqxvnlZYIctoHFwMovw"}

        data = req.form["data"]

        maxL = int(req.form["maxL"])
        minL = maxL // 4

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": data,
            "parameters": {"min_length": minL, "max_length": maxL},
        })[0]

        file2 = open("document.txt","w+",encoding="utf-8")
        file2.write(output["summary_text"])
        file2.close()


        return render_template("textresult.html", result=output["summary_text"],orignaldata=data)
    else:
        return render_template("index.html")





@app.route('/lang')
def lang():
    return render_template("index2.html")

@app.route('/language',methods=['POST','GET'])
def language():
   if req.method == "POST":
     model2= pickle.load(open('modelcan.pkl', 'rb'))

     data = req.form["data" ]

     features2 = [data]
     print(features2)

     prediction2 = model2.predict(features2)
     output = prediction2

     return render_template("langdetector.html", result=output,sentence=data)


   else:
       return render_template("index2.html")


@app.route('/Down')
def Down():
    p="document.txt"
    return send_file(p,as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)