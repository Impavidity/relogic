from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/')
def index():
  return jsonify("Hello World")

class Server(object):
  def __init__(self, trainer=None):
    self.trainer = trainer

  def start(self):
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
  server = Server()
  server.start()