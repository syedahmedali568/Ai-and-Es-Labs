#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

# Initialize recognizer
recording = sr.Recognizer()

# Use microphone for input
with sr.Microphone() as source:
    print("Please Say something (With Noise):")
    # Listen to the audio without adjusting for noise
    audio = recording.listen(source)

# Convert audio to numpy array for plotting
audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

# Plot the waveform for "with noise"
plt.figure(figsize=(10, 4))
plt.plot(audio_data)
plt.title('Audio Waveform (With Noise)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Try to recognize the speech (what you said)
try:
    print("You said (With Noise): \n" + recording.recognize_google(audio))
except Exception as e:
    print("Error recognizing speech: ", e)


# In[84]:


import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

# Initialize recognizer
recording = sr.Recognizer()

# Use microphone for input
with sr.Microphone() as source:
    # Adjust for ambient noise to reduce noise impact
    recording.adjust_for_ambient_noise(source)
    print("Please Say something (Without Noise):")
    # Listen to the audio
    audio = recording.listen(source)

# Convert audio to numpy array for plotting
audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

# Plot the waveform for "without noise"
plt.figure(figsize=(10, 4))
plt.plot(audio_data)
plt.title('Audio Waveform (Without Noise)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Try to recognize the speech (what you said)
try:
    print("You said (Without Noise): \n" + recording.recognize_google(audio))
except Exception as e:
    print("Error recognizing speech: ", e)


# In[ ]:




