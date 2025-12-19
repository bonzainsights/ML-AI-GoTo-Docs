from flask import Flask

app = Flask(__name__)

from app import routes
app.register_blueprint(routes.main)
