1_review_data_for_prediction -

Code to generate business_id and review_text for food establishements in Toronto 'review_text_data_generator.py'

Summary -
Input1 - business.json (From Yelp Dataset)
Input2 - review.json (From Yelp Dataset)
Output - Records with clean business_id, text 

How to execute code: $ spark-submit review_text_data_generator.py path/business.json path/review.json path/output

---------------------------------------------------------
2_train_data_model_results

In this we applied machine learning to predict the sentiment of a yelp restaurant review based on three features - Price , Service and Food.
We used FastAI and Transfer Learning techniques to be able to predict the sentiment of a yelp review.

ULMFIT_model_train.py -

ULMFit, is an efficient transfer learning architecture which the authors claim can perform efficiently with relatively small training data. 
Our model used  discriminative fine tuning, and gradual unfreezing on an AWD-LSTM model, using three different labeled datasets. 

Creating a Labeled Dataset for Fine-Tuning Pre-trained ULMFIT Model. 
We identified three features that would be interesting for businesses: Food,Pricing and Service. 
In order to train our model we did not have a labeled dataset, we created a labeled dataset of 1122 reviews rated using 1- positive, 2-neutral, and 3- negative on the three features independently.

Dataset 1 food_label_train_data_yelp.csv 
Dataset 2 price_label_train_data_yelp.csv 
Dataset 3 service_label_train_data_yelp.csv

Output - Modeled output files are saved in the same folder. (TRAING RESULTS FOOD, TRAING RESULTS PRICE and TRAINING RESULTS SERVICE)

How to execute code: $ python ULMFIT_model_train.py path food_label_data_yelp.csv 

Import Required Libraries: pandas, numpy, matplotlib, nltk, PyTorch and fastai v1. The model is setup to automatically run on GPU, which is recommended.

Training will need to be run independently three time for Dataset 1, Dataset 2 and Dataset 3, the model will be saved, the user will need to save the model files for each using a different file name. 

-------------------------------------------------------------
3_prediction_generator - Generate the predictions based on given input text data.

predict_ULMFIT_model_food.py
predict_ULMFIT_model_price.py
predict_ULMFIT_model_service.py

How to execute code: $ python predict_ULMFIT_model_food.py food_label_data_yelp.csv {input text files with column0: businessID/reviewID, column1: text}
                     $ python predict_ULMFIT_model_price.py price_label_data_yelp.csv {input text files with column0: businessID/reviewID, column1: text}
                     $ python predict_ULMFIT_model_service.py service_label_data_yelp.csv {input text files with column0: businessID/reviewID, column1: text}
                     
From running Step 2(train_data_model_results), the model file will be saved in the operating environment in folder: model
This prediction will need to be run independently three times for food, price and service and the output is saved as csv
The input text files are formated and created using Step1: review_text_data_generator.py

-------------------------------------------------------------
4_final_combined_tables

We combined output files which are generated in step 3 -Price, Food,Service. Also we combined them with Business and Vader in order to visualize the final results. We dumped all final results into Cassandra tables.

Input Files - 
input1 - business.csv, 
input2 - review.csv, 
input3 - food.csv, 
input4 - service.csv, 
input5-price.csv 
input6 - vader.json
Cassandra key space - amahadev
Table names and keyspace hardcoded in the code which can be changed to the user requirements.


-------------------------------------------------------------
5 vader_json_language_filter.py

We used joint file of Business and Review Datasets as input for the Vader and we applied the fast.ai and VADER sentiment analyzer for the sentiment analysis.
The output of sentiment analyzer scores indexed seperately. The result of having seperate scores allowed the file to join the "Sentiment scores business" program.

How to execute code: $ python vader_json_language_filter.py bus_review.json path/output

Output consisted business_id, negative,positive,neutral and composite score of sentiment analysis.

---------------------------------------------------------------------
6 yelp_vis.py

This code used to create html page to visualize overall sentiment analysis. It has html link for Tabelau visulization also.


