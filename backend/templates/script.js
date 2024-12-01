const backendURL = '/process-image';

const imageUpload = document.getElementById('imageUpload');
const recognizedText = document.getElementById('recognizedText');

imageUpload.addEventListener('change', async function () {
    const file = imageUpload.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById('imagePreview').src = e.target.result;
            document.querySelector('.image-preview-container').style.display = 'block';
            document.getElementById('fileName').textContent = file.name;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(backendURL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to process the image');
            }

            const result = await response.json();
            recognizedText.value = result.recognized_text;
        } catch (error) {
            console.error('Error:', error);
            recognizedText.value = 'Error occurred.';
        }
    }
});




// Custom Cursor Handling
const customCursor = document.getElementById('customCursor');

// Show cursor on mouse enter and hide on mouse leave
document.addEventListener('mouseenter', () => {
    customCursor.classList.remove('hidden');
});
document.addEventListener('mouseleave', () => {
    customCursor.classList.add('hidden');
});

// Update cursor position on mouse move
document.addEventListener('mousemove', (e) => {
    customCursor.style.left = `${e.pageX}px`;
    customCursor.style.top = `${e.pageY}px`;
});

// Add "Active" Effect on Click
document.addEventListener('mousedown', () => {
    customCursor.classList.add('active');
});
document.addEventListener('mouseup', () => {
    customCursor.classList.remove('active');
});
