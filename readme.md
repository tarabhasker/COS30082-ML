1. Important folders in main directories
--Backend--
- dataset = stores plantvillage dataset and PV_train.csv
- model = stores weights of trained models

--Frontend--
- contains code for frontend

--model-notebooks--
- contains the baselin and new approach notebooks used to train the models


2. This project implements three models:
- Model 1 : Visual Language Clip
- Model 2 : Dual Branch
- Model 3 : Dual Branch with Domain Discriminator

3. Steps to run:
1. Navigate to backend and run "python app.py"
2. Navigate to frontend and run "http-server -c-1 --cors"
3. Navigate to the page displayed in the frontend. Example:
        Available on:
        http://127.0.0.1:8080
        http://192.168.4.53:8080
        Hit CTRL-C to stop the server
4. Choose the preferred model and upload an image to view results 

**Upon initial loading, installation of certain modules that your system lacks may be required.