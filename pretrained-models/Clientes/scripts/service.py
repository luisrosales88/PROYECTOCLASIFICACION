#Import Flask
from flask import Flask, request
from keras.preprocessing import image
from cnn_executor import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - IDENTIFICAR CLIENTES DE ALTO VALOR!'

@app.route('/clientes/', methods=['GET','POST'])
def rayosx():
	return 'MODELO IDENTIFICAR CLIENTES DE ALTO VALOR!'

@app.route('/clientes/default/', methods=['GET','POST'])
def default():
	#print (request.args)
	
	data = None
	if request.method == 'GET':
		print ("GET Method")
		data = request.args

	if request.method == 'POST':
		print ("POST Method")
		if (request.is_json):
			data = request.get_json()

	print("Data received:", data)
	
	# Obteniendo parametros
	saldo = data.get("saldo")
	estado = data.get("estado")
	nroEntidades = data.get("nroEntidades")
	saldoTotal = data.get("saldoTotal")
	saldoMN = data.get("saldoMN")
	saldoME = data.get("saldoME")
	lineaTC = data.get("lineaTC")
	utilizadoTC = data.get("utilizadoTC")
	entidadesNoReguladas = data.get("entidadesNoReguladas")
	ultimoMonto = data.get("ultimoMonto")
	ultimaTasa = data.get("ultimaTasa")
	nroCreditosVigentes = data.get("nroCreditosVigentes")
	nroCreditosCancelados = data.get("nroCreditosCancelados")
	nroCreditosCastigados = data.get("nroCreditosCastigados")
	
	cliente = np.array([[saldo,estado,nroEntidades,saldoTotal,saldoMN,saldoME,lineaTC,utilizadoTC,entidadesNoReguladas,ultimoMonto,ultimaTasa,nroCreditosVigentes,nroCreditosCancelados,nroCreditosCastigados]])
	
	scload=load('../model/std_scaler.bin')
	
	cliente = scload.transform(cliente)
	
	with graph.as_default():
		resultado = ""
		score = loaded_model.predict(cliente)
		print("\nFinal score: ", score)
		abandona = (score > 0.5)
		if abandona:
			resultado += "Es cliente de alto valor"
		else:
		    resultado += "No es cliente de alto valor"
		return resultado + ', score: ' + str(score[0])

# Run de application
app.run(host='0.0.0.0',port=5000)

# http://35.225.31.46:5000/clientes/default/?saldo=0.0381469622561374&estado=0.166666666666667&nroEntidades=0.142857142857143&saldoTotal=0.0510709922089954&saldoMN=0.0510792191104299&saldoME=0&lineaTC=0&utilizadoTC=0&entidadesNoReguladas=0&ultimoMonto=0.0594356613968381&ultimaTasa=0.453197637894208&nroCreditosVigentes=0.25&nroCreditosCancelados=0.0555555555555556&nroCreditosCastigados=0
