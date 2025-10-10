document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const predictBtn = document.getElementById('predict-btn');
    const imagePreview = document.getElementById('image-preview');
    const resultsArea = document.getElementById('results');

    // Preview uploaded image
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Process prediction request
    predictBtn.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }

        // Display loading state
        resultsArea.innerHTML = '<p>Recognizing...</p>';

        // Build FormData
        const formData = new FormData();
        formData.append('image', file);

        try {
            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                resultsArea.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                // Display results
                displayResults(data.results);
            }
        } catch (error) {
            resultsArea.innerHTML = `<p style="color: red;">Request failed: ${error.message}</p>`;
        }
    });

    // Display recognition results
    function displayResults(results) {
        let html = '<h3>Recognition Results (Top 3):</h3>';
        results.forEach((result, index) => {
            const probability = (result.probability * 100).toFixed(2);
            html += `
                <div class="result-item">
                    <div class="result-class">${index + 1}. ${result.class}</div>
                    <div class="result-prob">Confidence: ${probability}%</div>
                </div>
            `;
        });
        resultsArea.innerHTML = html;
    }
});