import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

#Load Data
#Loading training data
with open('train_qa.txt','rb') as f:
    train_data = pickle.load(f)
    
#Loading test data
with open('test_qa.txt','rb') as f:
    test_data = pickle.load(f)
    
#Data Preparation
all_data = train_data + test_data # Combine train data and test data
def vocab_creator(data):
    '''
    Creating vocabulary of words present in dataset
    '''
    vocab = set()
    for story, question, answer in data:
        vocab = vocab.union(set(story))
        vocab = vocab.union(set(question))
    vocab.add('yes')
    vocab.add('no')
    
    tokenizer = Tokenizer(filters = [])
    tokenizer.fit_on_texts(vocab)
    
    max_story_len = max([len(datum[0]) for datum in data]) #maximum story length
    max_question_len = max([len(datum[1]) for datum in data]) #maximum question length
    
    return tokenizer, max_story_len, max_question_len, vocab
    
tokenizer, max_story_len, max_question_len, vocab = vocab_creator(all_data) 
vocab_size = len(vocab) + 1 #size of our vocabulary
input_sequence = Input((max_story_len,)) #placeholder for shape = (max_story_len, batch_size)
question = Input((max_question_len,))


def vectorize_stories(data, word_index = tokenizer.word_index, 
                      max_story_len = max_story_len, 
                      max_question_len = max_question_len):
    '''
    Vectorizes story, question, and answer with padded sequences
    X = Stories
    X_Q = Questions
    Y  = Answers
    '''
    X  = []
    X_Q = []
    Y = []
    
    for stories ,questions, answers in data:
        '''
        For each story
        [23,14,43,...]
        '''
        x = [word_index[word.lower()] for word in stories]
        x_q = [word_index[word.lower()] for word in questions]
        
        y = np.zeros(len(word_index) + 1)
        
        y[word_index[answers]] = 1
        
        X.append(x)
        X_Q.append(x_q)
        Y.append(y)
    return (pad_sequences(X, maxlen = max_story_len), pad_sequences(X_Q, maxlen = max_question_len), np.array(Y))
    
    
inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

#Data Modeling
def model(vocab_size = vocab_size, max_question_len = max_question_len, input_sequence = input_sequence, question = question):
    '''
    Input Encoder M
    Gets embedded to a seq of vecotrs
    '''
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim = 64))
    input_encoder_m.add(Dropout(0.3))

    '''
    Input Encoder C
    Gets embedded to a seq of vecotrs
    '''
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim = max_question_len))
    input_encoder_c.add(Dropout(.3))
    
    '''
    Question Encoder
    '''
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim = 64, input_length = max_question_len))
    question_encoder.add(Dropout(.3))
    
    '''
    ENCODED RESULT <-- ENCODER(INPUT)
    '''
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded =  question_encoder(question)

    '''
    Use dot product to compute the match between the first input vector
    seq and the query
    '''
    match = dot([input_encoded_m, question_encoded], axes =(2,2))
    match = Activation('softmax')(match)
    
    '''
    Add above match matrix to second input vector seq
    '''
    response = add([match, input_encoded_c])
    response = Permute((2,1))(response) # convert it to have a output of samples dim by query_max_len by story_max_len
    
    '''
    Now we can concat the match matrix with the question vector seq
    '''
    answer = concatenate([response, question_encoded])
    answer = LSTM(32)(answer)
    answer = Dropout(0.5)(answer)
    answer = Dense(vocab_size)(answer) # (sample, vocab_size)
    answer = Activation('softmax')(answer)
    model = Model([input_sequence, question], answer)
    model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
# Run model
model = model()
model.summary()


# Training data
history = model.fit([inputs_train, queries_train], answers_train, batch_size = 32, epochs = 40, validation_data=([inputs_test, queries_test], answers_test))
