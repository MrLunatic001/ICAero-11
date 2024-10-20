

function updateOutput() {

        // Get the latitude and longitude input values
        const latitude = document.getElementById('latitudeInput').value;
        const longitude = document.getElementById('longitudeInput').value;

        // Make sure latitude and longitude are valid
        if (latitude && longitude) {
                // Update the iframe src dynamically to point to the input coordinates
                const mapIframe = document.getElementById('mapIframe');
                const googleMapUrl = `https://www.google.com/maps/embed/v1/view?key=AIzaSyBrLaXm88LtgVDha_eMR3ZxvJHbPKVs8QU&center=${latitude},${longitude}&zoom=14&maptype=satellite`;
                mapIframe.src = googleMapUrl;

                // Optionally, update the output text
                document.getElementById('output').innerText = `Displaying map for Latitude: ${latitude}, Longitude: ${longitude}`;
        } else {
                document.getElementById('output').innerText = "Please enter both latitude and longitude.";
        }
}

