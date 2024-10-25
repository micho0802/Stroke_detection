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

app.post('/predict', upload.single('imagefile'), async (req, res) => {
    const image_path = req.file.path;
    const {
        age, hypertension, heart_disease, ever_married, work_type,
        residence_type, avg_glucose_level, bmi, heart_rate, smoking_status
    } = req.body;

    try {
        // Create a FormData instance
        const formData = new FormData();
        
        // Append the image file to the form data (reading the file from disk)
        formData.append('imagefile', fs.createReadStream(req.file.path));

        // Append the medical history data to the form data
        formData.append('age', age);
        formData.append('hypertension', hypertension);
        formData.append('heart_disease', heart_disease);
        formData.append('ever_married', ever_married);
        formData.append('work_type', work_type);
        formData.append('residence_type', residence_type);
        formData.append('avg_glucose_level', avg_glucose_level);
        formData.append('bmi', bmi);
        formData.append('heart_rate', heart_rate);
        formData.append('smoking_status', smoking_status);

        // Send the form data to the Flask server
        const predictionResponse = await axios.post('http://localhost:3003/predict', formData, {
            headers: formData.getHeaders()
        });

        const predictionResult = predictionResponse.data;  // Get prediction result object with stroke_probability and no_stroke_probability

        // Insert the image path, message, and prediction into MongoDB
        await collection.insertMany([{ image_path, msg, prediction: predictionResult }]);

        // Send the result back to the frontend
        res.json(predictionResult);  // Return the prediction in JSON format
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
