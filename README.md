# IT Ticket Classification
> In IT support environments, efficiently handling large volumes of incoming tickets is crucial for reducing response time and improving customer satisfaction. This project aims to automate the classification of IT support tickets into predefined categories using Natural Language Processing (NLP) and Deep Learning. By analyzing ticket descriptions, the goal is to categorize them accurately, improving response times and optimizing resource allocation. The project not only utilizes traditional deep learning models but also extends the approach by leveraging Llama 3.1, a state-of-the-art large language model (LLM), to enhance classification performance. This extension demonstrates the ability to apply cutting-edge LLMs in real-world applications, showcasing their potential in handling complex text data and improving overall operational efficiency.


## Table of Contents
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)



## General Information
- IT support teams receive large volumes of tickets daily, making manual classification inefficient and sometimes incorrect. Misclassification of tickets can lead to delays in resolving critical issues. The objective is to build a ticket classification model that can automatically assign tickets to predefined categories. The project compares multiple models to identify the most effective approach.
- The project starts with a traditional machine learning approach, involving TFIDF Vectorization, followed by RandomForest.
- For further improvement, GloVe word embedding technique was applied, which is a powerful pretrained word embedding to capture semantic meaning in the text data. This was followed by applying and tuning deep learning models like GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory). After training and evaluation the LSTM model was seen to provide most decent performance.
- for LSTM model with droput, train accuracy was 0.9270, test accuracy was 0.9130, test F1 score was 0.9131.
  
- The second part of the project aims to apply Llama 3.1 (meta-llama/Meta-Llama-3.1-8B-Instruct), which is a state of the art large Language Model for performing the classification task. Due to constraints, 24% of the data was used for the purpose and model was optimized to consume less time while maintaining decent performance. The goal was to demonstrate how the LLM models can handle complex text data efficiently and at scale.
- PEFT-based fine-tuning using LoRA was applied to tune the LLM model.
- With only a subset of the data, accuracy was 0.8947 and F1 score was 0.8946 suggesting a strong potential for the model. With more data and resources, it is expected that the performance would further improve, solidifying the practical application of LLMs for text classification.


## Conclusions
Several approaches were evaluated to arrive at the final model. The initial method involved TFIDF Vectorization and traditional machine learning models like RandomForest. The deep learning approach incorporated Tokenization and the GloVe word embedding technique, followed by models like GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory). The extension of this project applies Llama 3.1, demonstrating the effectiveness of large language models in text classification tasks. Despite using a subset of the full dataset, the LLM-based model shows promising results.

LSTM (Long Short-Term Memory) with Dropout

- Train accuracy: 0.9270
- Test accuracy: 0.9130
- Test F1 score: 0.9131

Llama 3.1 (meta-llama/Meta-Llama-3.1-8B-Instruct) trained with 24% of the data

- Accuracy: 0.8947
- F1 score: 0.8946


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
