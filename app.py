from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return 'API de IA activa'

@app.route('/api/prediccion', methods=['POST'])
def prediccion():
    print("hola, prueba")
