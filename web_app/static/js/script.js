// Глобальные переменные
let currentFile = null;
let webcamStream = null;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    initUpload();
    initWebcam();
});

// Управление вкладками
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');

            // Скрыть все вкладки
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Показать выбранную вкладку
            button.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');

            // Скрыть результаты при переключении вкладок
            document.getElementById('results-section').style.display = 'none';
        });
    });
}

// Инициализация загрузки файлов
function initUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const analyzeButton = document.getElementById('analyze-button');
    const newImageButton = document.getElementById('new-image-button');
    const resultsNewImageButton = document.getElementById('results-new-image');

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Выбор файла через кнопку
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Анализ изображения
    analyzeButton.addEventListener('click', () => {
        if (currentFile) {
            analyzeImage(currentFile);
        }
    });

    if (newImageButton) {
        newImageButton.addEventListener('click', () => {
            startNewUpload(true);
        });
    }

    if (resultsNewImageButton) {
        resultsNewImageButton.addEventListener('click', () => {
            startNewUpload(true);
        });
    }

    function startNewUpload(openDialog) {
        const uploadTabButton = document.querySelector('.tab-button[data-tab="upload"]');
        if (uploadTabButton && !uploadTabButton.classList.contains('active')) {
            uploadTabButton.click();
        }

        currentFile = null;
        previewImage.src = '';
        previewSection.style.display = 'none';
        uploadArea.style.display = 'block';
        document.getElementById('results-section').style.display = 'none';

        if (openDialog) {
            fileInput.value = '';
            fileInput.click();
        }
    }

    // Обработка выбора файла
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Пожалуйста, выберите изображение');
            return;
        }

        currentFile = file;

        // Показать превью
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewSection.style.display = 'block';
            document.getElementById('results-section').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

// Анализ изображения
async function analyzeImage(file) {
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // Показать загрузку
    loadingOverlay.style.display = 'flex';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Ошибка при анализе изображения');
        }

        const result = await response.json();

        // Скрыть загрузку
        loadingOverlay.style.display = 'none';

        // Показать результаты
        displayResults(result);
        resultsSection.style.display = 'block';

        // Прокрутка к результатам
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        loadingOverlay.style.display = 'none';
        alert('Ошибка: ' + error.message);
    }
}

// Отображение результатов
function displayResults(result) {
    // Основная эмоция
    document.getElementById('result-emoji').textContent = result.emoji;
    document.getElementById('result-emotion').textContent = result.emotion_ru;

    // Уверенность
    const confidence = Math.round(result.confidence * 100);
    document.getElementById('confidence-value').textContent = confidence + '%';
    document.getElementById('confidence-fill').style.width = confidence + '%';
    document.getElementById('confidence-fill').style.background =
        `linear-gradient(90deg, ${result.color} 0%, ${result.color}dd 100%)`;

    // Обработанное изображение
    if (result.processed_image) {
        document.getElementById('processed-image').src = result.processed_image;
    }

    // Статус обнаружения лица
    const faceStatus = result.face_detected ?
        '✓ Лицо обнаружено и проанализировано' :
        '⚠ Лицо не обнаружено. Анализ всего изображения';
    document.getElementById('face-detection-status').textContent = faceStatus;

    // График вероятностей
    // График вероятностей
    if (result.all_probabilities && typeof result.all_probabilities === 'object') {
        displayProbabilities(result.all_probabilities);
    }

}

// Отображение графика вероятностей
function displayProbabilities(probabilities) {
    const chartContainer = document.getElementById('probabilities-chart');
    chartContainer.innerHTML = '';

    const emotions = {
        'Счастье': { emoji: '😊', color: '#FFD93D' },
        'Грусть': { emoji: '😢', color: '#6BCB77' },
        'Нейтральное': { emoji: '😐', color: '#C7CEEA' }
    };


    // Сортировка по убыванию вероятности
    const sortedEmotions = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedEmotions.forEach(([emotion, probability]) => {
        const percent = Math.round(probability * 100);
        const emotionData = emotions[emotion];

        const barElement = document.createElement('div');
        barElement.className = 'probability-bar';
        barElement.innerHTML = `
            <div class="probability-label">
                ${emotionData.emoji} ${emotion}
            </div>
            <div class="probability-bar-container">
                <div class="probability-bar-fill" style="width: ${percent}%; background: ${emotionData.color};">
                    ${percent > 10 ? percent + '%' : ''}
                </div>
            </div>
            <div class="probability-value">${percent}%</div>
        `;

        chartContainer.appendChild(barElement);
    });
}

// Инициализация веб-камеры
function initWebcam() {
    const startButton = document.getElementById('start-webcam');
    const captureButton = document.getElementById('capture-photo');
    const stopButton = document.getElementById('stop-webcam');
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');

    startButton.addEventListener('click', async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480
                }
            });

            video.srcObject = webcamStream;

            startButton.style.display = 'none';
            captureButton.style.display = 'inline-block';
            stopButton.style.display = 'inline-block';

        } catch (error) {
            alert('Не удалось получить доступ к камере: ' + error.message);
        }
    });

    captureButton.addEventListener('click', () => {
        // Захват кадра
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0);

        // Конвертация в blob
        canvas.toBlob(async (blob) => {
            const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
            await analyzeImage(file);
        }, 'image/jpeg');
    });

    stopButton.addEventListener('click', () => {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;

            startButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
            stopButton.style.display = 'none';
        }
    });
}

// Утилиты
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}
