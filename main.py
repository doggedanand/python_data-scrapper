from flask import Flask, request, jsonify
import pdf_worker
import scrape_pdf
import pike_pdf_worker
# init flask app
app = Flask(__name__)
# home route
@app.route("/")
def home():
    return "Hello, Flask! Welcome to the application."
# pdf upload route
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    try:
        # debug log
        print("Received PDF data of length:")
        # get raw pdf bytes
        pdf_data = request.data  
        # debug log length
        print("Received PDF data of length:", len(pdf_data))
        # check pdf exists
        if not pdf_data:
            return jsonify({"error": "No PDF received"}), 400
        # success response
        return jsonify({"message": "PDF processed successfully", "result": []})
    except Exception as e:
        # error response
        return jsonify({"error": str(e)}), 500
# run app
if __name__ == "__main__":
    app.run(debug=True)
