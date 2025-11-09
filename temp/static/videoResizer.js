document.addEventListener('DOMContentLoaded', () => {
    const videoStream = document.getElementById('video-stream');
    const zoomInButton = document.getElementById('zoom-in');
    const zoomOutButton = document.getElementById('zoom-out');

    if (!videoStream || !zoomInButton || !zoomOutButton) {
        return;
    }

    let currentWidth = 85;
    const step = 5;
    const minWidth = 40;
    const maxWidth = 100;

    // --- NEW FUNCTION to check width and disable/enable buttons ---
    const updateButtonStates = () => {
        // Disable the '+' button if the width is at or above the max
        zoomInButton.disabled = currentWidth >= maxWidth;
        // Disable the '-' button if the width is at or below the min
        zoomOutButton.disabled = currentWidth <= minWidth;
    };

    videoStream.style.width = `${currentWidth}%`;

    zoomInButton.addEventListener('click', () => {
        if (currentWidth < maxWidth) {
            currentWidth += step;
            videoStream.style.width = `${currentWidth}%`;
            updateButtonStates(); // Call the update function
        }
    });

    zoomOutButton.addEventListener('click', () => {
        if (currentWidth > minWidth) {
            currentWidth -= step;
            videoStream.style.width = `${currentWidth}%`;
            updateButtonStates(); // Call the update function
        }
    });

    // --- Call the function once on load to set the initial state ---
    updateButtonStates();
});