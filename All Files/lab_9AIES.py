#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/KhuzaimaHassan/AI-and-ES/blob/main/lab_9AIES.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import nltk
nltk.download('punkt')



# In[ ]:


from nltk.tokenize import sent_tokenize
text="Hello, world! How's everything going? All good here"
sentences=sent_tokenize(text)
print(sentences)



# In[ ]:


from nltk.tokenize import word_tokenize
text="Hello, world! How's everything going? I can’t wait for the weekend."
words=word_tokenize(text)
print(words)


# In[ ]:


from nltk.tokenize import WordPunctTokenizer

# Input text
text = "Tokenization is challenging."

# Character Tokenization
char_tokens = list(text.replace(" ", ""))
print("Character Tokens:", char_tokens)

# Subword Tokenization using WordPunctTokenizer
tokenizer = WordPunctTokenizer()
subword_tokens = tokenizer.tokenize(text)
print("Subword Tokens:", subword_tokens)


# In[ ]:


from transformers import AutoTokenizer

# Load tokenizer (BERT tokenizer example)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Input text
text = "Tokenization is challenging."

# Character Tokenization
char_tokens = list(text.replace(" ", ""))
print("Character Tokens:", char_tokens)

# Subword Tokenization
subword_tokens = tokenizer.tokenize(text)
print("Subword Tokens:", subword_tokens)


# In[ ]:


import random
import nltk

# Download NLTK data for tokenization and preprocessing (only required once)
nltk.download('punkt')

def tokenize_input(user_input):
    # Tokenize the user input into words
    return nltk.word_tokenize(user_input.lower())

# Expanded responses for different intents
greetings = [
    "Hello! How can I assist you today?",
    "Hi there! What can I do for you?",
    "Hello! How's it going?",
    "Hey! How are you?",
    "Greetings! How can I help you?",
    "Hi! Hope you're having a great day!"
]

farewells = [
    "Goodbye! Have a great day!",
    "See you later!",
    "Goodbye! Take care!",
    "Bye! Looking forward to our next chat.",
    "Farewell! Let me know if you need help again.",
    "Catch you later! Stay safe!"
]

unknown_responses = [
    "I'm not sure how to respond to that.",
    "Could you please clarify?",
    "I didn't understand that.",
    "I'm here to help, but I didn't quite catch that.",
    "Hmm, I'm not sure. Could you rephrase?",
    "I’m not entirely sure. Could you try asking differently?"
]

# Responses for "Five Ws" questions
who_responses = [
    "I'm just a simple bot created to chat with you!",
    "I'm your friendly virtual assistant here to help.",
    "I'm a chatbot, designed to make your day a bit easier.",
    "I'm an AI developed to answer your questions and keep you company."
]

what_responses = [
    "I can help answer questions, have a chat, or keep you company.",
    "I'm here to assist you with whatever I can.",
    "I'm a virtual assistant here to lend a hand with information or advice.",
    "I help with small tasks, conversation, and general queries."
]

when_responses = [
    "I'm available anytime you need!",
    "Whenever you'd like! I'm here 24/7.",
    "I'm here round the clock, so feel free to reach out anytime.",
    "Anytime works for me! Just type, and I’ll be here."
]

where_responses = [
    "I'm right here with you, wherever you are!",
    "I'm in the digital world, ready to help wherever you may be.",
    "I exist in the virtual realm, always ready to chat.",
    "Wherever you have an internet connection, that’s where I’ll be!"
]

why_responses = [
    "That's a great question! I was created to chat and assist with basic information.",
    "I'm here to make your day a bit easier and answer your questions!",
    "I was designed to be your helpful companion for questions and small tasks.",
    "Why? Because chatting with people like you is what I’m here for!"
]

def get_response(user_input):
    tokens = tokenize_input(user_input)

    # Greeting intent
    if any(word in tokens for word in ["hello", "hi", "hey", "greetings", "hola", "hi there"]):
        return random.choice(greetings)

    # Farewell intent
    elif any(word in tokens for word in ["bye", "goodbye", "see you", "farewell", "later"]):
        return random.choice(farewells)

    # Weather-related intent
    elif "weather" in tokens:
        return "I'm not connected to the internet, so I can't give you a real weather update, but it's always sunny when we're chatting!"

    # Who-related intent
    elif "who" in tokens:
        return random.choice(who_responses)

    # What-related intent
    elif "what" in tokens:
        return random.choice(what_responses)

    # When-related intent
    elif "when" in tokens:
        return random.choice(when_responses)

    # Where-related intent
    elif "where" in tokens:
        return random.choice(where_responses)

    # Why-related intent
    elif "why" in tokens:
        return random.choice(why_responses)

    # Unknown input
    else:
        return random.choice(unknown_responses)

# Main loop to interact with the user
def chat():
    print("Bot: Hello! I'm here to chat with you. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Bot:", random.choice(farewells))
            break

        response = get_response(user_input)
        print("Bot:", response)

# Start chatting
chat()

