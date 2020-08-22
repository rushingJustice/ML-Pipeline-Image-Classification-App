Computer Vision <> Online App Example
1. State Farm kaggle competition
2. re-download driver images
3. Do resnet 50 from Keras (see example from predict_dog Medium article)
4. Containerize app in either dockeerhub or gci. 
5. App UI: Input image of driver, get prediction. Will need to research how to upload image.
6. Expose to internet


curl -X POST -F image=@dog.jpg "http://localhost:5000/predict"

