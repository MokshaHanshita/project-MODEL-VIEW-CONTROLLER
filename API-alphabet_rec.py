from flask import Flask,jsonify,request
from Alphabet_rec import get_prediction

app=Flask(__name__)
@app.route('/predict-ALpa',methods=['POST'])

def predict_ALpha():
    img=request.files.get("Alphabet")
    Alphabet=get_prediction(img)
    return jsonify({
        'Status':'Scuscess',
        'prediction':Alphabet[0]
    })
app.run(debug=True)