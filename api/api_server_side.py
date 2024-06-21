from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import os
from api_preprocess import find_the_number, classify_number

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist

@app.route('/')
def index():
    # Simple HTML template for the upload form with JavaScript for popup
    return render_template_string('''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image</h1>
    <form id="uploadForm" method="POST" action="/upload" enctype="multipart/form-data">
      <input type="file" name="file" id="fileInput">
      <input type="submit" value="Upload">
    </form>
    
    <script>
      document.getElementById('uploadForm').onsubmit = function(event) {
          event.preventDefault();  // Prevent the default form submission
          
          var formData = new FormData(document.getElementById('uploadForm'));

          // Show popup window
          alert('Image is being uploaded!');

          fetch('/upload', {
              method: 'POST',
              body: formData
          }).then(response => response.json())
            .then(data => {
                alert('Number on image is: ' + data.prediction);
                console.log(data);
            })
            .catch(error => {
                alert('Upload failed!');
                console.error(error);
            });
      };
    </script>
    ''')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    img = Image.open(file.stream)
    
    # Define the path where the image will be saved
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    img.save(file_path)
    prediction = classify_number(find_the_number(file_path))
    return jsonify({'status': 'success', 'file_path': file_path, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
