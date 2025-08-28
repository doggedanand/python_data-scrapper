from flask import Flask
import pdf_worker
import scrape_pdf
# import pike_pdf_worker
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask! Welcome to the application."

if __name__ == "__main__":
    app.run(debug=True)