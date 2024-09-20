const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const multer = require('multer');  // Add multer
const collection = require('./mongo');

const app = express();
const templatePath = path.join(__dirname, '../frontend/templates');
const staticPath = path.join(__dirname, '../frontend/public');

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Set up storage for multer
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join(__dirname, '../frontend/public/uploads')); // Ensure this directory exists
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname)); // Save file with unique name
    }
});

const upload = multer({ storage: storage });

// Set view engine and views path
app.set('view engine', 'hbs');
app.set('views', templatePath);

// Serve static files (CSS, JS, images)
app.use(express.static(staticPath));

// Routes
app.get('/', (req, res) => {
    res.render('home');
});

// Handle form submission and image upload
app.post('/predict', upload.single('imagefile'), async (req, res) => {
    const imagePath = '/uploads/' + req.file.filename;  // Path where the image is stored
    const msg = req.body.msg;

    // Ensure you are saving the image path to the correct schema field
    await collection.insertMany([{ msg, image_path: imagePath }]);

    res.send("Data saved with image");
});

// Connect to MongoDB (uncomment this when ready)
// mongoose.connect('mongodb://localhost:27017/UMKC_DS_Capstone');

app.listen(3002, () => {
    console.log("Server running on port 3002");
});
