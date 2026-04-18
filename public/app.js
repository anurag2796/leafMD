const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const previewImage = document.getElementById('preview-image');
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');
const detectionsList = document.getElementById('detections-list');
const loadingState = document.getElementById('loading-state');

// Botanical Dictionary
const botanicalDB = {
    // TOMATO
    "Tomato_Bacterial_spot": { severity: "High", cause: "Bacterial (Xanthomonas spp.)", symptoms: "Small, dark, greasy spots on leaves; fruit lesions.", treatment: "Use copper bactericides. Remove infected debris." },
    "Tomato_Early_blight": { severity: "Medium", cause: "Fungal (Alternaria solani)", symptoms: "Dark, irregular spots with concentric rings on lower leaves.", treatment: "Apply fungicides (Chlorothalonil). Ensure adequate air circulation." },
    "Tomato_Late_blight": { severity: "High", cause: "Oomycete (Phytophthora infestans)", symptoms: "Large soaked spots on leaves; white mold on undersides.", treatment: "Immediate fungicide application. Destroy infected plants." },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": { severity: "High", cause: "Viral (TYLCV, Whitefly vector)", symptoms: "Upward curling leaves, yellowing margins, stunted growth.", treatment: "Control whiteflies. Use insecticidal soaps. No cure for infected plants." },
    "Tomato__Tomato_mosaic_virus": { severity: "High", cause: "Viral (ToMV)", symptoms: "Mottled light/dark green leaves, fern-like stunted growth.", treatment: "Remove infected plants immediately. Sanitize tools." },
    "Tomato_Leaf_Mold": { severity: "Medium", cause: "Fungal (Passalora fulva)", symptoms: "Pale green spots on upper leaves, olive-green mold underneath.", treatment: "Reduce humidity. Ensure spacing. Copper fungicides." },
    "Tomato_Septoria_leaf_spot": { severity: "Medium", cause: "Fungal (Septoria lycopersici)", symptoms: "Small, circular spots with dark borders on older leaves.", treatment: "Apply fungicidal sprays. Avoid overhead watering." },
    "Tomato_Spider_mites_Two_spotted_spider_mite": { severity: "High", cause: "Pest (Tetranychus urticae)", symptoms: "Yellow stippling on leaves, fine webbing under leaves.", treatment: "Apply horticultural oils or predatory mites. Regular misting." },
    "Tomato_healthy": { severity: "Low", cause: "None", symptoms: "Vibrant green foliage, robust stems.", treatment: "Maintain healthy soil, regular fertilization, and steady watering." },

    // POTATO
    "Potato___Early_blight": { severity: "Medium", cause: "Fungal (Alternaria solani)", symptoms: "Brown target-like concentric rings on older leaves.", treatment: "Apply protective fungicides before symptoms spread." },
    "Potato___Late_blight": { severity: "High", cause: "Oomycete (Phytophthora infestans)", symptoms: "Water-soaked lesions on leaves, rapid rot.", treatment: "Immediate destruction of infected matter. Apply specific systemic fungicides." },
    "Potato___healthy": { severity: "Low", cause: "None", symptoms: "Healthy green leaves, firm stems.", treatment: "Maintain good cultivation practices." },

    // PEPPER
    "Pepper__bell___Bacterial_spot": { severity: "High", cause: "Bacterial (Xanthomonas)", symptoms: "Water-soaked spots turning dark, defoliation.", treatment: "Use copper-based sprays. Avoid overhead irrigation." },
    "Pepper__bell___healthy": { severity: "Low", cause: "None", symptoms: "Lush, unmarked green leaves.", treatment: "Continue standard NPK fertilization regimen." }
};

// Default fallback
const defaultBotany = { severity: "Medium", cause: "Unknown Pattern", symptoms: "Detected anomalies matching visual signatures.", treatment: "Isolate plant and consult local agricultural extension." };

// Drag and drop event listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a visual image file.');
        return;
    }

    // Set preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        
        // Setup UI state
        dropZone.style.display = 'none';
        resultContainer.classList.remove('hidden');
        loadingState.classList.remove('hidden');
        detectionsList.innerHTML = '';
        
        // Wait for image to load to set canvas size
        previewImage.onload = () => {
            canvas.width = previewImage.width;
            canvas.height = previewImage.height;
            uploadImage(file);
        };
    };
    reader.readAsDataURL(file);
}

async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const useClahe = document.getElementById('toggle-clahe').checked;
    const useTta = document.getElementById('toggle-tta').checked;
    const apiUrl = `/detect?use_clahe=${useClahe}&use_tta=${useTta}`;

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        renderResults(data.predictions);
    } catch (error) {
        console.error('Error:', error);
        loadingState.classList.add('hidden');
        detectionsList.innerHTML = `<li style="color:var(--danger)">Error: ${error.message}</li>`;
    }
}

function renderResults(predictions) {
    loadingState.classList.add('hidden');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factor since previewImage display size might differ from intrinsic size
    const scaleX = previewImage.width / previewImage.naturalWidth;
    const scaleY = previewImage.height / previewImage.naturalHeight;

    if (predictions.length === 0) {
        detectionsList.innerHTML = `<li class="detection-item"><span class="detection-name">No diseases detected</span><span class="detection-score">✅</span></li>`;
        return;
    }

    predictions.forEach(pred => {
        const { x1, y1, x2, y2 } = pred.box;
        
        // Scale bounding box coordinates to UI scale
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;

        // Draw bounding box
        ctx.strokeStyle = '#2ed573';
        ctx.lineWidth = 3;
        ctx.strokeRect(sx1, sy1, width, height);

        // Prevent label clipping at the top
        const labelY = (sy1 < 30) ? Math.max(sy1 + height + 25, 25) : sy1;

        // Draw Label Background
        ctx.fillStyle = '#2ed573';
        ctx.font = '16px Outfit, sans-serif';
        const text = `${pred.class_name.replace(/_/g, ' ')} ${(pred.confidence * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(sx1, labelY - 25, textWidth + 10, 25);

        // Draw Label Text
        ctx.fillStyle = '#0b1110';
        ctx.fillText(text, sx1 + 5, labelY - 7);

        // Build Detailed Card
        const li = document.createElement('li');
        li.className = 'diagnosis-card';
        
        const cleanName = pred.class_name.replace(/_/g, ' ').replace('healthy', 'Healthy');
        const botany = botanicalDB[pred.class_name] || defaultBotany;
        
        li.innerHTML = `
            <div class="diagnosis-header">
                <span class="diagnosis-name">${cleanName}</span>
                <span class="severity-${botany.severity}">${botany.severity} Severity</span>
            </div>
            <div class="botanical-info">
                <div class="info-row">
                    <span class="info-label">Confidence:</span>
                    <span>${(pred.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Pathogen:</span>
                    <span>${botany.cause}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Symptoms:</span>
                    <span>${botany.symptoms}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Treatment:</span>
                    <span>${botany.treatment}</span>
                </div>
            </div>
        `;
        detectionsList.appendChild(li);
    });
}

function resetUI() {
    dropZone.style.display = 'block';
    resultContainer.classList.add('hidden');
    fileInput.value = ''; // Reset input
    ctx.clearRect(0, 0, canvas.width, canvas.height); // clear canvas
}
