from flask import Flask, current_app


app = Flask(__name__)


# rota da pagina principal

@app.route('/')
def homepage():
    return current_app.send_static_file('index.html')

# rotas alternativas


@app.route("/app")
def aplication():
    return current_app.send_static_file("rota1.html")


@app.route("/app/imagem")  # rota a definir no futuro
def resposta():
    return current_app.send_static_file("rota2.html")


if __name__ == "__main__":
    app.run(threaded=True, port=8081, debug=True)


# utilizar post pra enviar uma imagem pro site esperando uma resposta de true ou false
# exemplo request.post(link, data=imagem)
