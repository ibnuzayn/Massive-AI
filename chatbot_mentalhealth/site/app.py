from flask import Flask, request, jsonify, render_template, session
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import LSTM # type: ignore
from tensorflow.keras.initializers import Orthogonal # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import ast

app = Flask(__name__)  
app.secret_key = 'root' # Ganti dengan kunci rahasia yang aman

# Custom LSTM layer class to filter out unsupported arguments
class CustomLSTM(LSTM):
    def __init__(self, units, **kwargs):
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(units, **kwargs)

# Load model with custom objects
custom_objects = {'Orthogonal': Orthogonal, 'LSTM': CustomLSTM}
model = load_model('./model/model_chatbot_8.h5', custom_objects=custom_objects)

# Load data
df = pd.read_csv('./data/data_chatbot_mental_health.csv')

# Ensure all values in the 'patterns' column are strings and handle NaN values
df['patterns'] = df['patterns'].astype(str).fillna('')

# Convert the 'responses' column from string to list
df['responses'] = df['responses'].apply(ast.literal_eval)

# Recreate tokenizer
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])  # Convert pattern texts to sequences of numbers
X = pad_sequences(ptrn2seq, padding='post')  # Pad sequences with zeros to make them of equal length

lbl_enc = LabelEncoder()  # Create a LabelEncoder object
y = lbl_enc.fit_transform(df['tag'])  # Convert class labels to numbers

# Load responses based on tags
responses_data = {}
for _, row in df.iterrows():
    responses_data[row['tag']] = row['responses']

# List of relevant tags to display
relevant_tags = ["scared", "anxious", "depressed", "suicide", "stressed", "worthless"]

# List of questions
pertanyaan = [
    "Selamat siang, apa kabar? Senang sekali bisa bertemu dengan Anda hari ini. Silakan jawab pertanyaan-pertanyaan berikut.<br>1. Bagaimana perasaan Anda secara umum belakangan ini?",
    "\n2. Apakah Anda sering merasa cemas atau gelisah tanpa alasan yang jelas?",
    "\n3. Apakah Anda sering merasa sedih atau kehilangan minat pada hal-hal yang biasanya Anda nikmati?",
    "\n4. Bagaimana pola tidur Anda belakangan ini? Apakah Anda sering kesulitan tidur atau tidur terlalu banyak?",
    "\n5. Apakah Anda merasa tertekan atau putus asa dalam situasi tertentu?",
    "\n6. Apakah Anda sering merasa terlalu lelah atau kehilangan energi?",
    "\n7. Apakah Anda sering merasa kesepian atau terisolasi dari orang lain?",
    "\n8. Apakah Anda sering memiliki pikiran atau keinginan untuk menyakiti diri sendiri?",
    "\n9. Apakah Anda merasa sulit untuk berkonsentrasi atau membuat keputusan?",
    "\n10. Bagaimana hubungan Anda dengan orang-orang terdekat Anda? Apakah Anda sering mengalami konflik atau kesulitan dalam berkomunikasi?",
    "\n11. Apakah Anda sering merasa tertekan oleh tuntutan pekerjaan atau sekolah?",
    "\n12. Apakah Anda memiliki pengalaman traumatis atau kenangan yang mengganggu?",
    "\n13. Bagaimana pola makan Anda belakangan ini? Apakah Anda sering kehilangan nafsu makan atau makan berlebihan?",
    "\n14. Apakah Anda merasa sulit untuk mengontrol emosi Anda atau sering merasa mudah tersinggung?",
    "\n15. Apakah Anda sering mengalami gejala fisik seperti sakit kepala, nyeri otot, atau gangguan pencernaan tanpa penyebab medis yang jelas?"
]

# Function to analyze user input and predict mental health issues
def analyze_input(user_input):
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, maxlen=24, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_label = lbl_enc.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction) * 100
    if predicted_label[0] in responses_data:
        response = np.random.choice(responses_data[predicted_label[0]])
        return response, predicted_label[0], confidence
    return "Maaf, saya tidak bisa memahami Anda.", None, confidence

# Function to analyze mental health conditions based on user responses
def analyze_mental_health(user_responses):
    masalah_kesehatan_mental = []
    for response in user_responses:
        sequences = tokenizer.texts_to_sequences([response])
        padded_sequences = pad_sequences(sequences, maxlen=24, padding='post')
        prediction = model.predict(padded_sequences)
        predicted_label = lbl_enc.inverse_transform([np.argmax(prediction)])
        if predicted_label[0] in relevant_tags and predicted_label[0] not in masalah_kesehatan_mental:
            masalah_kesehatan_mental.append(predicted_label[0])

    analysis_result = []
    if masalah_kesehatan_mental:
        for issue in masalah_kesehatan_mental:
            analysis_result.append({
                "issue": issue,
                "response": np.random.choice(responses_data[issue]) if issue in responses_data else ""
            })
    return analysis_result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    if 'question_index' not in session:
        session['question_index'] = 0
        session['responses'] = []

    if user_input.lower() == 'temanngobrol':
        session['question_index'] = 0
        session['responses'] = []
        response = pertanyaan[session['question_index']]
        session['question_index'] += 1
        return jsonify({'response': response, 'is_questions': True})
    elif user_input.lower() == 'selesai':
        analysis = analyze_mental_health(session['responses'])
        session.pop('question_index', None)
        session.pop('responses', None)
        return jsonify({'response': analysis, 'is_questions': False})
    else:
        session['responses'].append(user_input)
        if session['question_index'] < len(pertanyaan):
            response = pertanyaan[session['question_index']]
            session['question_index'] += 1
            return jsonify({'response': response, 'is_questions': True})
        else:
            response, tag, confidence = analyze_input(user_input)
            return jsonify({'response': response, 'is_questions': False})

if __name__ == '__main__':
    app.run()
