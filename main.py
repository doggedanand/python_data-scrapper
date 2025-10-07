from flask import Flask, json, request, jsonify 
import os
# import pdf_worker
from scrape_pdf import handle_uploaded_pdf
# import pike_pdf_worker
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
        file_data = request.files['pdf']  
         # 2. get "setting" metadata (JSON string)
        setting_raw = request.form.get("setting")
        # debug log length
        # print("Received PDF data of length:", (file))
        # check pdf exists
        if not file_data:
            return jsonify({"error": "No PDF received"}), 400
        if not setting_raw:
            return jsonify({"error": "No setting provided"}), 400
        setting = json.loads(setting_raw)
        # print("Received setting:", setting)
        # 3. save file to disk (or process in memory)
        save_path = os.path.join("uploads", file_data.filename)
        # print('save path', save_path)
        os.makedirs("uploads", exist_ok=True)
        file_data.save(save_path)
        # Process the PDF with settings
        result = handle_uploaded_pdf(save_path, setting)
        # print("Processing result:", result)
        # Remove the temporary file
        os.remove(save_path)
        # success response
        return jsonify({"message": "PDF processed successfully", "result": result or {}}), 200
    except Exception as e:
        print("exception error", e)
        # error response
        return jsonify({"error": str(e)}), 500
    finally:
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception as cleanup_error:
                # Optionally log this error
                print(f"Failed to delete temporary file: {cleanup_error}")
# run app
if __name__ == "__main__":
    app.run(debug=True)
