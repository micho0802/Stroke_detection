# Multimodal for Stroke Detection

This repository provides the code for our extension on applied machine learning in healthcare, particularly Multimodal for Stroke Detection. We provide an example of how our platform works with medical history and facial analysis.

## Dataset

The training dataset consists of 2 different source datasets. 

- Facial stroke data: https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke
- Medical history data: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset


## Architecture
Like most capstone projects, the platform involves frontend, backend, and database architecture.

### Frontend:
- The frontend was built around [Gradio](https://www.gradio.app/), a fast API for implementing pre-trained multimodal models in production.

### Backend:
- The backend technology was construct arround JavaScript. 
### Database:
- The database used is [MongoDB](https://www.mongodb.com/), a sufficent database for unstructure data.

### Multimodal architecture
The machine learning implementation including 2 approach:
- Medical history prediction:
  - Where various ML models will be testing on medical history dataset.
- Facial detection:
  - Where various CNN architectures will be implement on facial expression.


## Result
![Result](../Result.PNG)


## Stroke prediction with face and medical history
![Screenshot from 2025-02-12 18-10-32](https://github.com/user-attachments/assets/589e83f9-5eea-45b3-9a90-892e345c8aec)




## Multimodal chatbot response
![Screenshot from 2025-02-12 18-12-01](https://github.com/user-attachments/assets/29ffa1f6-3507-4fdf-8444-2f096d940b2b)



## Reference
[1] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105–6114.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[3] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[4] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.

[5] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

[6] A. Dosovitskiy et al., "A ConvNet for the 2020s," arXiv preprint arXiv:2201.03545, 2022.

[7] Meta AI, "The LLaMA 3 Herd of Models," Meta AI Blog, Apr. 2024. [Online]. Available: https://ai.meta.com/blog/meta-llama-3/

