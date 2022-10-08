from flask import Flask, current_app
app = Flask(__name__)

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
