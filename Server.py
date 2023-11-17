from flask import Flask, request
import Sentiment
import pandas as pd
app = Flask(__name__)


@app.route("/")
def hello_word():
    return "<p>Hello world</p>"

@app.route("/exo", methods=['POST'])
def verify():
    data = request.get_json()
    data = data['data']
    response = Sentiment.Predict(data)
    if(response[0] == 2):
        return("Neutral")
    if(response[0] == 1):
        return("Negative")
    else:
        return("Positive")
  

if __name__ == "__main__":
    app.run()

