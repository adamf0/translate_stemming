from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory='')

ket_tada = 'kata tidak ada dalam datasheet'
ket_ada = 'kata dasar ada dalam datasheet'

class Translator:
   
    def __init__(self):
        self.latent_dim = 256
        self.source_path = 'model/source_words'
        self.target_path = 'model/target_words'
        self.modelpath = 'model/seq2seq.h5'
        self.max_encoder_seq_length = 16
        self.max_decoder_seq_length = 59
        self.input_characters = [item for item in open(self.source_path).read().split('*')]
        self.target_characters = [item for item in open(self.target_path).read().split('*')]
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

    
    def load_model(self):
        model = load_model(self.modelpath)
        encoder_inputs = model.input[0]   # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_inputs = model.input[1]   # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

  
    def decode_sequence(self, input_seq):
        encoder_model, decoder_model = self.load_model()
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, len(self.target_characters)))
        target_seq[0, 0, self.target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, len(self.target_characters)))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]

        return decoded_sentence

    
    def encode_sentence(self, input_text):
        encode_input = np.zeros((1, self.max_encoder_seq_length, len(self.input_characters)), dtype='float32')
        for index, char in enumerate(input_text):
            print(index, char)
            encode_input[0, index, self.input_token_index[char]] = 1.
        return encode_input

@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/stemming', response_class=HTMLResponse)
async def stemming(request: Request):
    return templates.TemplateResponse('stemming.html', {'request': request})

@app.post('/stemming/proses', response_class=HTMLResponse)
async def stemming_proses(request: Request, data: str = Form()):
    status: str = ket_tada

    hasil_pref = hilangkanPref(data)
    if hasil_pref[1]==ket_ada:
        status = hasil_pref[1]
    print("hasil_pref: ", json.dumps(hasil_pref[0]))
    
    hasil_suf = hilangkanSuf(hasil_pref[0])
    if hasil_suf[1]==ket_ada:
        status = hasil_suf[1]
    print("hasil_suf: ", json.dumps(hasil_suf[0]))

    hasil_sisipan = hilangkanSisipan(hasil_suf[0])
    if hasil_sisipan[1]==ket_ada:
        status = hasil_sisipan[1]
    print("hasil_sisipan: ", json.dumps(hasil_sisipan[0]))

    hasil_sisipan2 = hilangkanSisipan2(hasil_sisipan[0])
    if hasil_sisipan2[1]==ket_ada:
        status = hasil_sisipan2[1]
    print("hasil_sisipan2: ", json.dumps(hasil_sisipan2[0]))

    hasil_sisipan3 = hilangkanSisipan3(hasil_sisipan2[0])
    if hasil_sisipan3[1]==ket_ada:
        status = hasil_sisipan3[1]
    print("hasil_sisipan3: ", json.dumps(hasil_sisipan3[0]))

    hasil_sisipan4 = hilangkanSisipan4(hasil_sisipan3[0])
    if hasil_sisipan4[1]==ket_ada:
        status = hasil_sisipan4[1]
    print("hasil_sisipan4: ", json.dumps(hasil_sisipan4[0]))

    hasil_sisipan6 = hilangkanSisipan6(hasil_sisipan4[0])
    if hasil_sisipan6[1]==ket_ada:
        status = hasil_sisipan6[1]
    print("hasil_sisipan6: ", json.dumps(hasil_sisipan6[0]))

    hasil_sisipan7 = hilangkanSisipan7(hasil_sisipan6[0])
    if hasil_sisipan7[1]==ket_ada:
        status = hasil_sisipan7[1]
    print("hasil_sisipan7: ", json.dumps(hasil_sisipan7[0]))

    return templates.TemplateResponse("hasil_stemming.html", {"request": request, "data": data, 'base': hasil_sisipan7[0], 'status': status})

@app.get('/translate', response_class=HTMLResponse)
async def translate(request: Request):
    return templates.TemplateResponse('translate.html', {'request': request})

@app.post('/translate/proses', response_class=HTMLResponse)
async def translate_proses(request: Request, data: str = Form()):
    translator = Translator()
    input_seq = translator.encode_sentence(data)
    decoded_sentence = translator.decode_sequence(input_seq)
    # decoded_sentence = ''
    return templates.TemplateResponse('hasil_translate.html', {'request': request, 'data':data, 'result': decoded_sentence})

##############################################################################################################
def kataSunda():
    with open(r'kamus_sunda.txt') as word_file:
        return set(word.strip().lower() for word in word_file)  

def ini_kata_sunda(word, kata_sunda):
    return word.lower() in kata_sunda

def hilangkanPref(word):
    prefs = ['ba', 'barang', 'di', 'ka', 'N', 'pa', 'pada', 'pang', 'para', 'per', 'pi' 'sa', 'sang', 'si', 'silih', 'sili', 'ti', 'ting', 'pating']
    nasal = ['nga', 'ng'] #ganti pakai k
    nasal2 = ['nge']
    nasal3 = ['m'] #jika huruf awalan m ganti dengan huruf b atau p bandingkan dengan kamus
    nasal3 = ['ny'] #jika huruf awalan ny ganti dengan huruf c atau s bandingkan dengan kamus
    nasal4 = ['n'] #jika huruf awalan n ganti dengan huruf t bandingkan dengan kamus
    kata_sunda = kataSunda()
    
    for pre in prefs:
        if  word.startswith(pre):
            hapusPref = word[len(pre):]
            if ini_kata_sunda(hapusPref,kata_sunda):
                return(hapusPref,ket_ada,'sukses')
    
    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSuf(word):
    sufs = ['an','eun','keun','na','ing','ning']
    kata_sunda = kataSunda()
    
    for suf in sufs:
        if  word.endswith(suf):
            hapusSufs = word[:-len(suf)]
            if ini_kata_sunda(hapusSufs,kata_sunda):
                return(hapusSufs,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan(word):
    konsonan = 'l'
    konsonan2 = 'al'
    kata_sunda = kataSunda()
    
    if  word.startswith(konsonan):
        hapusPre = word[len(konsonan):]
        if hapusPre.startswith(konsonan2):
            gabungPre = hapusPre.replace(konsonan2,'l')
            if ini_kata_sunda(gabungPre,kata_sunda):
                return(gabungPre,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan2(word):
    konsonan = 'r'
    konsonan2 = 'al'
    kata_sunda = kataSunda()
    
    if  word.endswith(konsonan):
        gabungPre = word.replace(konsonan2,'')
        if ini_kata_sunda(gabungPre,kata_sunda):
            return(gabungPre,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan3(word):
    konsonan = ['tr', 'br', 'cr', 'kr', 'pr', 'jr', 'dr']
    konsonan2 = 'al'
    kata_sunda = kataSunda()

    for inf in konsonan:
        if  word.find(inf):
            hapusInf = word.replace(konsonan2,'')
            if ini_kata_sunda(hapusInf,kata_sunda):
                return(hapusInf,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan4(word):
    konsonan = 'ar'
    kata_sunda = kataSunda()

    for inf in konsonan:
        if  word.find(inf):
            hapusInf = word.replace(konsonan,'')
            if ini_kata_sunda(hapusInf,kata_sunda):
                return(hapusInf,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan6(word):
    konsonan = 'in'
    kata_sunda = kataSunda()

    for inf in konsonan:
        if  word.find(inf):
            hapusInf = word.replace(konsonan,'')
            if ini_kata_sunda(hapusInf,kata_sunda):
                return(hapusInf,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan7(word):
    konsonan = 'um'
    kata_sunda = kataSunda()

    for inf in konsonan:
        if  word.find(inf):
            hapusInf = word.replace(konsonan,'')
            if ini_kata_sunda(hapusInf,kata_sunda):
                return(hapusInf,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')