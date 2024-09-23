const mongoose = require('mongoose');

// Define the schema for stroke detection data
const schema = new mongoose.Schema({
    image_path: {
        type: String,
        required: true  // Ensure that the image path is required
    },
    msg: {
        type: String,
        required: false  // Optional message
    }
});

// Create the collection model
const collection = mongoose.model('stroke_detection_data', schema);

module.exports = collection;
