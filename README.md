# Multimodal for Stroke Detection


## 1. Abstract  
- Motivation: In 2022, the CDC reported that 1 in 6 deaths (17.5%) from cardiovascular disease in the U.S. was due to stroke. Every 40 seconds, someone in the U.S. has a stroke; every 3 minutes and 11 seconds, someone dies from it. Patients who arrive at the ER within three hours of symptom onset experience less disability three months after a stroke. Immediate action in stroke cases greatly improves survival rates and outcomes. 

- Problem Statement: Stroke is a critical medical emergency that requires prompt treatment, but many stroke victims face delays in care due to late recognition of symptoms and challenges in accessing emergency services. This lack of timely intervention remains a significant barrier to improving stroke outcomes.

- Proposal: Leveraging recent advancements in AI and machine learning, we propose a platform for early stroke detection and fast emergency response. This platform uses AI-driven facial analysis and medical history data to detect potential stroke symptoms, alert users, and direct them to nearby emergency facilities, ensuring faster intervention and better patient outcomes.


## 2. Dataset
- Facial dataset: The dataset comprises 5029 images across two classes—acute stroke and non-stroke cases. Data augmentation techniques, such as flipping, rotation, and scaling, were applied to enhance model accuracy by diversifying and strengthening the dataset to more accurately reflect real-world scenarios.

![image](https://github.com/user-attachments/assets/2229fa54-4d43-4d7a-9524-a5051a980424)

- Medical history dataset: This dataset contains 5110 observations with 12 features, and is used to predict whether a patient is likely to get a stroke based on input parameters like gender, age, various diseases, and smoking status

![image](https://github.com/user-attachments/assets/6728ed3f-4120-4372-b2de-e9378587a543)
