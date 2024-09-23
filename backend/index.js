const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');  // Import the form-data package
const fs = require('fs');  // File system to read the uploaded file
const collection = require('./mongo');
const app = express();

// Set up Multer to handle file uploads
const upload = multer({ dest: 'uploads/' });

// Set the path to your frontend templates
const templatePath = path.join(__dirname, '../frontend/templates');

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.set('view engine', 'hbs');
app.set('views', templatePath);

// Serve the home page
app.get('/', (req, res) => {
    res.render('home');
});

// Handle the POST request to upload an image and send it to Flask for prediction
app.post('/sendMsg', upload.single('imagefile'), async (req, res) => {
    const image_path = req.file.path;
    const msg = req.body.msg;

    try {
        // Create a FormData instance
        const formData = new FormData();
        
        // Append the file to the form data (reading the file from disk)
        formData.append('imagefile', fs.createReadStream(req.file.path));

        // Send the form data to the Flask server
        const predictionResponse = await axios.post('http://localhost:3003/predict', formData, {
            headers: formData.getHeaders()
        });

        const predictionResult = predictionResponse.data.prediction;  // Get prediction result

        // Insert the image path, message, and prediction into MongoDB
        await collection.insertMany([{ image_path, msg, prediction: predictionResult }]);

        // Render the result page with the prediction
        res.render('result', { result: predictionResult });
    } catch (error) {
        console.error("Error processing the request:", error);
        res.status(500).send("Error processing the request: " + error.message);
    }
});

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/UMKC_DS_Capstone')
.then(() => console.log("MongoDB connected"))
.catch(err => {
    console.error("Failed to connect to MongoDB", err);
    process.exit(1);
});

// Start the Express server
app.listen(3002, () => {
    console.log("Port 3002 connected");
});
