DZone 

Modern API Management: Dive into APIs’ growing influence across domains, prevalent paradigms, microservices, the role AI plays, and more.

Download the NEW Trend Report
DZone Data Engineering AI/ML Build Your First Python Chatbot Project
Build Your First Python Chatbot Project
In this article, see a tutorial on how to build a Python chatbot project.
By Shivashish Thkaur user avatar Shivashish Thkaur DZone Core CORE · Updated May. 04, 20 · Tutorial
 Like (31)
 Comment (12)
 Save
  Tweet 
  Share 
 138.8K Views
Laptop with code on screen

Introduction
Chatbots are extremely helpful for business organizations and also the customers. The majority of people prefer to talk directly from a chatbox instead of calling service centers. Facebook released data that proved the value of bots. More than 2 billion messages are sent between people and companies monthly. The HubSpot research tells us that 71% of people want to get customer support from messaging apps. It is a quick way to get their problems solved so chatbots have a bright future in organizations.

Today we are going to build an exciting project on Chatbot. We will implement a chatbot from scratch that will be able to understand what the user is talking about and give an appropriate response.

Prerequisites
To implement the chatbot, we will be using Keras, which is a Deep Learning library, NLTK, which is a Natural Language Processing toolkit, and some helpful libraries. Run the below command to make sure all the libraries are installed:

pip install tensorflow keras pickle nltk 

If you want to learn Python for free, then here is the Master guide to learn Python for free. 

You may also like:   Build It Yourself: Chatbot API With Keras/TensorFlow Model
How do Chatbots Work?
Chatbots are nothing but an intelligent piece of software that can interact and communicate with people just like humans. Interesting, isn’t it? So now let's see how they actually work.

All chatbots come under the NLP (Natural Language Processing) concepts. NLP is composed of two things:

NLU (Natural Language Understanding): The ability of machines to understand human language like English.

NLG (Natural Language Generation): The ability of a machine to generate text similar to human written sentences.

Imagine a user asking a question to a chatbot: “Hey, what’s on the news today?”

The chatbot will break down the user sentence into two things: intent and an entity. The intent for this sentence could be get_news as it refers to an action the user wants to perform. The entity tells specific details about the intent, so "today" will be the entity. So this way, a machine learning model is used to recognize the intents and entities of the chat.

Project File Structure
After the project is complete, you will be left with all these files. Lets quickly go through each of them. It will give you an idea of how the project will be implemented.

Train_chatbot.py — In this file, we will build and train the deep learning model that can classify and identify what the user is asking to the bot.

Gui_Chatbot.py — This file is where we will build a graphical user interface to chat with our trained chatbot.

Intents.json — The intents file has all the data that we will use to train the model. It contains a collection of tags with their corresponding patterns and responses.

Chatbot_model.h5 — This is a hierarchical data format file in which we have stored the weights and the architecture of our trained model.

Classes.pkl — The pickle file can be used to store all the tag names to classify when we are predicting the message.

Words.pkl — The words.pkl pickle file contains all the unique words that are the vocabulary of our model.

Download the source code and the dataset: https://drive.google.com/drive/folders/1r6MrrdE8V0bWBxndGfJxJ4Om62dJ2OMP?usp=sharing



How to Build Your Own Chatbot
I’ve simplified the building of this chatbot in 5 steps:

Step 1. Import Libraries and Load the Data
Create a new python file and name it as train_chatbot and then we are going to import all the required modules. After that, we will read the JSON data file in our Python program.

Python

1
import numpy as np
2
from keras.models import Sequential
3
from keras.layers import Dense, Activation, Dropout
4
from keras.optimizers import SGD
5
import random
6
7
import nltk
8
from nltk.stem import WordNetLemmatizer
9
lemmatizer = WordNetLemmatizer()
10
import json
11
import pickle
12
13
intents_file = open('intents.json').read()
14
intents = json.loads(intents_file)




Step 2. Preprocessing the Data
The model cannot take the raw data. It has to go through a lot of pre-processing for the machine to easily understand. For textual data, there are many preprocessing techniques available. The first technique is tokenizing, in which we break the sentences into words.

By observing the intents file, we can see that each tag contains a list of patterns and responses. We tokenize each pattern and add the words in a list. Also, we create a list of classes and documents to add all the intents associated with patterns.

Python

1
words=[]
2
classes = []
3
documents = []
4
ignore_letters = ['!', '?', ',', '.']
5
6
for intent in intents['intents']:
7
    for pattern in intent['patterns']:
8
        #tokenize each word
9
        word = nltk.word_tokenize(pattern)
10
        words.extend(word)        
11
        #add documents in the corpus
12
        documents.append((word, intent['tag']))
13
        # add to our classes list
14
        if intent['tag'] not in classes:
15
            classes.append(intent['tag'])
16
17
print(documents)




Another technique is Lemmatization. We can convert words into the lemma form so that we can reduce all the canonical words. For example, the words play, playing, plays, played, etc. will all be replaced with play. This way, we can reduce the number of total words in our vocabulary. So now we lemmatize each word and remove the duplicate words.

Python

1
# lemmaztize and lower each word and remove duplicates
2
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
3
words = sorted(list(set(words)))
4
# sort classes
5
classes = sorted(list(set(classes)))
6
# documents = combination between patterns and intents
7
print (len(documents), "documents")
8
# classes = intents
9
print (len(classes), "classes", classes)
10
# words = all words, vocabulary
11
print (len(words), "unique lemmatized words", words)
12
13
pickle.dump(words,open('words.pkl','wb'))
14
pickle.dump(classes,open('classes.pkl','wb'))




In the end, the words contain the vocabulary of our project and classes contain the total entities to classify. To save the python object in a file, we used the pickle.dump() method. These files will be helpful after the training is done and we predict the chats.

Step 3. Create Training and Testing Data
To train the model, we will convert each input pattern into numbers. First, we will lemmatize each word of the pattern and create a list of zeroes of the same length as the total number of words. We will set value 1 to only those indexes that contain the word in the patterns. In the same way, we will create the output by setting 1 to the class input the pattern belongs to.

Python

1
# create the training data
2
training = []
3
# create empty array for the output
4
output_empty = [0] * len(classes)
5
# training set, bag of words for every sentence
6
for doc in documents:
7
    # initializing bag of words
8
    bag = []
9
    # list of tokenized words for the pattern
10
    word_patterns = doc[0]
11
    # lemmatize each word - create base word, in attempt to represent related words
12
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
13
    # create the bag of words array with 1, if word is found in current pattern
14
    for word in words:
15
        bag.append(1) if word in word_patterns else bag.append(0)
16
        
17
    # output is a '0' for each tag and '1' for current tag (for each pattern)
18
    output_row = list(output_empty)
19
    output_row[classes.index(doc[1])] = 1
20
    training.append([bag, output_row])
21
# shuffle the features and make numpy array
22
random.shuffle(training)
23
training = np.array(training)
24
# create training and testing lists. X - patterns, Y - intents
25
train_x = list(training[:,0])
26
train_y = list(training[:,1])
27
print("Training data is created")




Step 4. Training the Model
The architecture of our model will be a neural network consisting of 3 dense layers. The first layer has 128 neurons, the second one has 64 and the last layer will have the same neurons as the number of classes. The dropout layers are introduced to reduce overfitting of the model. We have used the SGD optimizer and fit the data to start the training of the model. After the training of 200 epochs is completed, we then save the trained model using the Keras model.save(“chatbot_model.h5”) function.

Python

1
# deep neural networds model
2
model = Sequential()
3
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
4
model.add(Dropout(0.5))
5
model.add(Dense(64, activation='relu'))
6
model.add(Dropout(0.5))
7
model.add(Dense(len(train_y[0]), activation='softmax'))
8
9
# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
10
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
11
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
12
13
#Training and saving the model 
14
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
15
model.save('chatbot_model.h5', hist)
16
17
print("model is created")




Step 5. Interacting With the Chatbot
Our model is ready to chat, so now let’s create a nice graphical user interface for our chatbot in a new file. You can name the file as gui_chatbot.py

In our GUI file, we will be using the Tkinter module to build the structure of the desktop application and then we will capture the user message and again perform some preprocessing before we input the message into our trained model.

The model will then predict the tag of the user’s message, and we will randomly select the response from the list of responses in our intents file.

Here’s the full source code for the GUI file.

Python

1
import nltk
2
from nltk.stem import WordNetLemmatizer
3
lemmatizer = WordNetLemmatizer()
4
import pickle
5
import numpy as np
6
7
from keras.models import load_model
8
model = load_model('chatbot_model.h5')
9
import json
10
import random
11
intents = json.loads(open('intents.json').read())
12
words = pickle.load(open('words.pkl','rb'))
13
classes = pickle.load(open('classes.pkl','rb'))
14
15
def clean_up_sentence(sentence):
16
    # tokenize the pattern - splitting words into array
17
    sentence_words = nltk.word_tokenize(sentence)
18
    # stemming every word - reducing to base form
19
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
20
    return sentence_words
21
# return bag of words array: 0 or 1 for words that exist in sentence
22
23
def bag_of_words(sentence, words, show_details=True):
24
    # tokenizing patterns
25
    sentence_words = clean_up_sentence(sentence)
26
    # bag of words - vocabulary matrix
27
    bag = [0]*len(words)  
28
    for s in sentence_words:
29
        for i,word in enumerate(words):
30
            if word == s: 
31
                # assign 1 if current word is in the vocabulary position
32
                bag[i] = 1
33
                if show_details:
34
                    print ("found in bag: %s" % word)
35
    return(np.array(bag))
36
37
def predict_class(sentence):
38
    # filter below  threshold predictions
39
    p = bag_of_words(sentence, words,show_details=False)
40
    res = model.predict(np.array([p]))[0]
41
    ERROR_THRESHOLD = 0.25
42
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
43
    # sorting strength probability
44
    results.sort(key=lambda x: x[1], reverse=True)
45
    return_list = []
46
    for r in results:
47
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
48
    return return_list
49
50
def getResponse(ints, intents_json):
51
    tag = ints[0]['intent']
52
    list_of_intents = intents_json['intents']
53
    for i in list_of_intents:
54
        if(i['tag']== tag):
55
            result = random.choice(i['responses'])
56
            break
57
    return result
58
59
#Creating tkinter GUI
60
import tkinter
61
from tkinter import *
62
63
def send():
64
    msg = EntryBox.get("1.0",'end-1c').strip()
65
    EntryBox.delete("0.0",END)
66
67
    if msg != '':
68
        ChatBox.config(state=NORMAL)
69
        ChatBox.insert(END, "You: " + msg + '\n\n')
70
        ChatBox.config(foreground="#446665", font=("Verdana", 12 )) 
71
72
        ints = predict_class(msg)
73
        res = getResponse(ints, intents)
74
        
75
        ChatBox.insert(END, "Bot: " + res + '\n\n')           
76
77
        ChatBox.config(state=DISABLED)
78
        ChatBox.yview(END)
79
80
root = Tk()
81
root.title("Chatbot")
82
root.geometry("400x500")
83
root.resizable(width=FALSE, height=FALSE)
84
85
#Create Chat window
86
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
87
88
ChatBox.config(state=DISABLED)
89
90
#Bind scrollbar to Chat window
91
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
92
ChatBox['yscrollcommand'] = scrollbar.set
93
94
#Create Button to send message
95
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
96
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
97
                    command= send )
98
99
#Create the box to enter message
100
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
101
#EntryBox.bind("<Return>", send)
102
103
#Place all components on the screen
104
scrollbar.place(x=376,y=6, height=386)
105
ChatBox.place(x=6,y=6, height=386, width=370)
106
EntryBox.place(x=128, y=401, height=90, width=265)
107
SendButton.place(x=6, y=401, height=90)
108
109
root.mainloop()



