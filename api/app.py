import json
from flask import Flask, request

app = Flask(__name__)

entities = []

@app.route("/app/entities", methods = ['GET', 'POST'])
def entitiesHandler():

    if request.method == 'GET':
        return {
            entities
        }
    
    if request.method == 'POST':
        
        print (request.data)
        print (type(request.data).__name__)
        entities.append(json.loads(request.data))

        return "dd"

    return "<p>Hello, World!</p>"