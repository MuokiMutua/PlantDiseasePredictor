# Plant Disease Detection
![image](https://github.com/user-attachments/assets/b9104321-be24-4d87-aa89-3f6f969af10c)

## Problem Statement
Plant diseases significantly impact agricultural productivity, leading to reduced yields and financial losses for farmers. Identifying diseases at an early stage can help in taking preventive measures and minimizing damage. However, manual disease identification is time-consuming and requires expert knowledge.

## Background
With the rise of machine learning and computer vision, deep learning models can now accurately classify plant diseases using images. By leveraging a trained model, farmers and agricultural professionals can diagnose diseases quickly and take necessary actions to prevent crop loss.

## Solution
This project utilizes a deep learning model trained on plant disease images to predict and classify different types of diseases. A Flask-based API is developed to process image inputs and return disease predictions, making it accessible via a simple web interface or API calls.

## Technologies Used
- **Python**: Primary programming language
- **Flask**: For building the API
- **TensorFlow/Keras**: Deep learning framework used for training the model
- **OpenCV**: Image preprocessing
- **Docker**: Containerization for deployment
- **JSON**: For mapping class indices to disease names
- **TOML**: Configuration management
- **Jupyter Notebook**: Training Model

## Project Structure
```
PLANT_DISEASE_DETECTION/
│── app/
│   ├── trained_model/
│   │   ├── plant_disease_pred_model.h5
│   │   ├── class_indices.json
│── config.toml
│── credentials.toml
│── Dockerfile
│── main.py
│── requirements.txt
│── venv/
```
## How It Works  
![image](https://github.com/user-attachments/assets/c5a928fd-5695-48d4-857e-2a8f983baa9f)

1. **User Uploads an Image**: A user uploads a picture of a plant leaf showing signs of disease.
2. ![image](https://github.com/user-attachments/assets/3d04bac3-81b2-411e-90f9-c90e82eff9d5)
 
3. **Preprocessing**: The image is resized and processed using OpenCV.  
4. **Model Prediction**: The trained deep learning model analyzes the image and classifies the disease.
5. ![image](https://github.com/user-attachments/assets/fba11411-f70a-48d4-bf34-7b3384ce5a52)

6. **Results Displayed**: The system returns the disease name along with possible treatments and prevention measures.  
7. **Deployment**: The application can be deployed as a web service using Flask and Docker.  

## Future Improvements
- Improve model accuracy with a larger dataset
- Develop a user-friendly web interface
- Deploy as a cloud-based service for accessibility

## License
This project is open-source under the MIT License.
