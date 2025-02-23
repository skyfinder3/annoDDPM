from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # only if you need cross-origin requests
import requests
import os
import subprocess
from PIL import Image
from io import BytesIO

DATASETS_DIR = "DATASETS/Analyze"
OUTPUT_DIR = "output_image"

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Enabling CORS if you're calling from a different origin
app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()

    if not data or "imageUrl" not in data or "argNr" not in data or "slice_id" not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    image_url = data["imageUrl"]
    argNr = str(data["argNr"])
    slice_id = str(data["slice_id"])

    try: 
        # Fetch the image from the provided URL
        response = requests.get(image_url)
        response.raise_for_status()

        # saving the file for the script to use
        filename = os.path.basename(image_url)
        image_path = os.path.join(DATASETS_DIR, filename)
        image = Image.open(BytesIO(response.content))
        with open(image_path, "wb") as img_file:
            img_file.write(response.content)

         # Run the external Python script
        output_path = "./output_image/final.png"
        subprocess.run(["python3", "analyze_image.py", argNr, image_path, slice_id], check=True)

       # Check if output file exists
        if not os.path.exists(output_path):
            return jsonify({"error": "Processed image not found"}), 500

        return send_file(output_path, mimetype="image/png")
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
    # Cleanup: delete temporary files after processing
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(output_path):
            os.remove(output_path)
    

@app.route("/", methods=['GET'])
def home():
    return "Up and Running"

@app.route('/message', methods=['GET'])
def funny_message():
    message = "Hi Robin, when are we going to get ramen and beer?"
    return jsonify({"funny_message": message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)

