async function predictDisease() {
    const imageUpload = document.getElementById('imageUpload');
    const resultSection = document.getElementById('resultSection');
    const predictedClass = document.getElementById('predictedClass');
    const confidence = document.getElementById('confidence');
    const matchedImages = document.getElementById('matchedImages');
  
    // Ensure an image is selected
    if (!imageUpload.files || imageUpload.files.length === 0) {
      alert('Please upload an image!');
      return;
    }
  
    // Create a FormData object to send the image to the backend
    const formData = new FormData();
    formData.append('file', imageUpload.files[0]);
  
    try {
      // Send the image to the backend API
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
  
      // Parse the JSON response
      const data = await response.json();
  
      // Update the UI with the prediction and matched images
      predictedClass.innerText = data.class;
      confidence.innerText = data.confidence;
  
      // Display matched images
      matchedImages.innerHTML = ""; // Clear previous images
      data.matchedImages.forEach((imgSrc) => {
        const img = document.createElement('img');
        img.src = imgSrc;
        matchedImages.appendChild(img);
      });
  
      // Show the result section
      resultSection.classList.remove('hidden');
    } catch (error) {
      console.error('Error predicting disease:', error);
      alert('An error occurred. Please try again later.');
    }
  }
  