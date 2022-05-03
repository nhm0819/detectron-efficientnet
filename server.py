from waitress import server
import app

server(app.app, host='0.0.0.0', port=80)