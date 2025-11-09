document.addEventListener('DOMContentLoaded', () => {
    const infoBox = document.getElementById('info-box');

    // Return early if the info-box element isn't on the page
    if (!infoBox) {
        return;
    }

    // Create an EventSource object to connect to the /data_feed route
    const eventSource = new EventSource("/data_feed");

    // This function is called every time a message is received from the server
    eventSource.onmessage = function(event) {
        try {
            // Parse the JSON string from the server
            const data = JSON.parse(event.data);

            // Update the content of the info-box with the new data
            // Using backticks (`) for a multi-line string makes this cleaner
            infoBox.innerHTML = `<h2>Live Data</h2>
                                 <p>
                                    Timestamp: ${data.timestamp}<br>
                                    Status: ${data.status}<br>
                                    Value: ${data.value.toFixed(2)}
                                 </p>`;
        } catch (error) {
            console.error("Failed to parse server data:", error);
        }
    };

    // Optional: Handle connection errors
    eventSource.onerror = function(err) {
        console.error("EventSource failed:", err);
        infoBox.innerHTML = `<h2>Live Data</h2><p>Connection to server lost.</p>`;
        eventSource.close(); // Close the connection on error
    };
});