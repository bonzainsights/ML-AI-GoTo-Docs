import sys
import os

# Add project directory to path for cPanel
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Attempt to import the Flask app
    # If this file is named app.py and there is a folder named app,
    # this specific import line is very likely to fail due to circular import/shadowing.
    from app import app as application
    app = application # Alias for compatibility
except Exception as e:
    import traceback
    trace = traceback.format_exc()
    def application(environ, start_response):
        status = '200 OK'
        output = f"App Startup Error:\n{trace}\n\nPath: {sys.path}\nCWD: {os.getcwd()}".encode('utf-8')
        response_headers = [('Content-type', 'text/plain'),
                            ('Content-Length', str(len(output)))]
        start_response(status, response_headers)
        return [output]
    app = application

if __name__ == "__main__":
    application.run()
