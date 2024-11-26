document.getElementById("image-upload").addEventListener("change", function () {
  const file = this.files[0];
  const uploadBox = document.querySelector(".upload-box");

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      // Set the uploaded image as the background
      uploadBox.style.backgroundImage = `url(${e.target.result})`;
      // Remove the gradient background and icon
      uploadBox.style.border = "none";
    };
    reader.readAsDataURL(file);
  }
});


document.getElementById("submit-btn").addEventListener("click", async function () {
  const imageUpload = document.getElementById("image-upload");
  const cropResult = document.getElementById("crop-result");
  const cropConfidence = document.getElementById("crop-confidence");
  const diseaseResult = document.getElementById("disease-result");
  const diseaseConfidence = document.getElementById("disease-confidence");

  if (!imageUpload.files || imageUpload.files.length === 0) {
    alert("Please upload an image!");
    return;
  }

  const formData = new FormData();
  formData.append("file", imageUpload.files[0]);

  try {
    const response = await fetch("http://localhost:5000/predict", { method: "POST", body: formData });
    const data = await response.json();

    // Update the results section
    cropResult.textContent = data.crop.name;
    cropConfidence.textContent = `${data.crop.class}%`;
    diseaseResult.textContent = data.disease.name;
    diseaseConfidence.textContent = `${data.disease.class}%`;

    // Fetch matching images for the carousel
    const imagesResponse = await fetch("http://localhost:5000/fetch-matching-images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ crop_label: data.crop.class, disease_label: data.disease.class }),
    });
    const imagesData = await imagesResponse.json();

    // Update carousel with fetched images
    const carouselImages = document.querySelector(".carousel-images");
    const leftButton = document.querySelector(".carousel-btn.left");
    const rightButton = document.querySelector(".carousel-btn.right");

    let images = imagesData.images.map(imagePath => `backend/dataset/plantvillage/${imagePath}`);
    let currentIndex = 0;

    // Function to update the displayed image
    function updateImage() {
      carouselImages.innerHTML = ""; // Clear previous image
      const img = document.createElement("img");
      img.src = images[currentIndex];
      img.alt = "Training Image";
      carouselImages.appendChild(img);

      // Disable buttons at the ends
      leftButton.disabled = currentIndex === 0;
      rightButton.disabled = currentIndex === images.length - 1;
    }

    // Event listeners for buttons
    leftButton.addEventListener("click", () => {
      if (currentIndex > 0) {
        currentIndex--;
        updateImage();
      }
    });

    rightButton.addEventListener("click", () => {
      if (currentIndex < images.length - 1) {
        currentIndex++;
        updateImage();
      }
    });

    // Initialize the carousel with the first image
    if (images.length > 0) {
      updateImage();
    } else {
      alert("No matching images found!");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred. Please try again.");
  }
});
