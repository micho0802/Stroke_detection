const mongoose = require('mongoose')
const { required } = require('yargs')

mongoose.connect("mongodb://localhost:27017/UMKC_DS_Capstone")
.then(() =>{
    console.log("Mongodb connected")
})
.catch(()=>{
    console.log("Failed")
})


const schema = new mongoose.Schema({
    image_path:{
        type:String,
        required:true
    }
})

const collection=new mongoose.model("stroke_detection_data", schema)

module.exports=collection
