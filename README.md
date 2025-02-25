# IT_Ticket_Classification_Project
> In IT support environments, handling large volumes of incoming tickets efficiently is crucial for reducing response time and improving customer satisfaction. This project aims to automate the classification of IT support tickets into predefined categories using Natural Language Processing (NLP) and Deep Learning. By analyzing ticket descriptions, the goal is to classify them into appropriate categories, improving response times and optimizing resource allocation.The aim is to make the model learn to understand textual descriptions and accurately predict the corresponding ticket category. This automation helps streamline ticket triaging, ensuring that issues are routed to the right teams faster and improving overall operational efficiency.


## Table of Contents
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)



## General Information
- IT support teams receive large volumes of tickets daily, making manual classification inefficient and sometimes incorrect. Misclassification of tickets can lead to delays in resolving critical issues. The objective is to build a ticket classification model that can automatically assign tickets to predefined categories. The project compares multiple models to identify the most effective approach.


## Conclusions
Several different approaches were tried and evaluated before arriving to the final model. The project involves an exploratory data analysis. The initial approach involves TFIDF Vectorization and fitting traditional machine learning models like Logistic Regression, RandomForest. This is followed by teh final approach which involves Tokenization, using a popular pretrained word embedding technique GloVe (Global Vectors for Word Representation). Deep Learning Models GRU (Gated Recurrent Unit) , LSTM (Long Short-Term Memory) were tried before arriving at teh final model.

Selected Model

LSTM (Long Short-Term Memory) with Dropout

- Train accuracy: 0.9270
- Test accuracy: 0.9130
- Test F1 score: 0.9131


## Technologies Used
- pandas - version 2.2.2
- tensorflow - version 2.18.0
- numpy - version 1.26.4
- matplotlib - version 3.7.1
- seaborn - version 0.13.2
- scikit-learn - version 1.3.1
- scipy - version 1.13.1



  ## Contact
  Anusha Chaudhuri [anusha761]
