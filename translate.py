from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

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


def test():
    en = 'belet'
    translator = Translator()
    input_seq = translator.encode_sentence(en)
    decoded_sentence = translator.decode_sequence(input_seq)
    print('-')
    print('Sunda :', en)
    print('Indo  :', decoded_sentence)

test()