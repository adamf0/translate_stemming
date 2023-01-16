from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import math
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
)

app = FastAPI()
templates = Jinja2Templates(directory='')

ket_tada = 'kata tidak ada dalam datasheet'
ket_ada = 'kata dasar ada dalam datasheet'

class Translator:
    def __init__(self):
        self.checkpoint = "facebook/nllb-200-distilled-600M" #nllb-200-distilled-1.3B 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.source = "sun_Latn"
        self.target = "ind_Latn"
        self.mode = "translation"
        self.length = 400 #400 max
        self.translator = pipeline(
            task = self.mode, 
            model = self.model, 
            tokenizer = self.tokenizer, 
            src_lang = self.source, 
            tgt_lang = self.target, 
            max_length = self.length
        )

    def translate(self, text):
        return self.translator(text)[0]['translation_text']

@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/stemming', response_class=HTMLResponse)
async def stemming(request: Request):
    return templates.TemplateResponse('stemming.html', {'request': request})

@app.post('/stemming/proses', response_class=HTMLResponse)
async def stemming_proses(request: Request, data: str = Form()):
    status: str = ket_tada
    kata_sunda = kataSunda()

    hasil_pref = hilangkanAwalan(data)
    print("hasil_pref: ", json.dumps(hasil_pref[0]))
    # if hasil_pref[1]==ket_ada:
    #     status = hasil_pref[1]
        
    hasil_suf = hilangkanSuf(hasil_pref[0])
    print("hasil_suf: ", json.dumps(hasil_suf[0]))
    # if hasil_suf[1]==ket_ada:
    #     status = hasil_suf[1]

    hasil_sisipan = hilangkanSisipan(hasil_suf[0])
    print("hasil_sisipan: ", json.dumps(hasil_sisipan[0]))
    # if hasil_sisipan[1]==ket_ada:
    #     status = hasil_sisipan[1]

    hasil_sisipan2 = hilangkanSisipan2(hasil_sisipan[0])
    print("hasil_sisipan2: ", json.dumps(hasil_sisipan2[0]))
    # if hasil_sisipan2[1]==ket_ada:
    #     status = hasil_sisipan2[1]

    hasil_sisipan3 = hilangkanSisipan3(hasil_sisipan2[0])
    print("hasil_sisipan3: ", json.dumps(hasil_sisipan3[0]))
    # if hasil_sisipan3[1]==ket_ada:
    #     status = hasil_sisipan3[1]

    hasil_sisipan4 = hilangkanSisipan4(hasil_sisipan3[0])
    print("hasil_sisipan4: ", json.dumps(hasil_sisipan4[0]))
    # if hasil_sisipan4[1]==ket_ada:
    #     status = hasil_sisipan4[1]

    hasil_sisipan6 = hilangkanSisipan6(hasil_sisipan4[0])
    print("hasil_sisipan6: ", json.dumps(hasil_sisipan6[0]))
    # if hasil_sisipan6[1]==ket_ada:
    #     status = hasil_sisipan6[1]

    hasil_sisipan7 = hilangkanSisipan7(hasil_sisipan6[0])
    print("hasil_sisipan7: ", json.dumps(hasil_sisipan7[0]))
    # if hasil_sisipan7[1]==ket_ada:
    #     status = hasil_sisipan7[1]

    hasil_barung = hilangkanBarung(hasil_sisipan7[0])
    print("hasil_barung: ", json.dumps(hasil_barung[0]))
    # if hasil_barung[1]==ket_ada:
    #     status = hasil_barung[1]
	
    hasil_bareng = hilangkanBareng(hasil_barung[0])
    print("hasil_bareng: ", json.dumps(hasil_bareng[0]))
    # if hasil_bareng[1]==ket_ada:
    #     status = hasil_bareng[1]   

    hasil_nasal = modulNasal(hasil_bareng[0])
    print("hasil_nasal: ", json.dumps(hasil_nasal[0]))
    # if hasil_nasal[1]==ket_ada:
    #     status = hasil_nasal[1]

    output = hasil_nasal[0]
    min_distance = math.inf
    most_similar = ""
    tmp = []

    if not ini_kata_sunda(output,kata_sunda):
        for string in kataSunda():
            distance = levenshtein_distance(output, string)
            if distance < min_distance:
                min_distance = distance
                most_similar = string
                tmp.append(string)
        print("String yang paling mendekati dengan '{}' adalah '{}' dengan jarak Levenshtein {}".format(output, most_similar, min_distance))
        tmp=tmp[::-1]

    print(json.dumps({"data": data, 'base': output, 'alternative': tmp, 'len': len(tmp)},sort_keys=True, indent=4))
    return templates.TemplateResponse("hasil_stemming.html", {"request": request, "data": data, 'base': output, 'alternative': tmp, 'len': len(tmp)})

@app.get('/translate', response_class=HTMLResponse)
async def translate(request: Request):
    return templates.TemplateResponse('translate.html', {'request': request})

@app.post('/translate/proses', response_class=HTMLResponse)
async def translate_proses(request: Request, data: str = Form()):
    # translator = Translator()
    # input_seq = translator.encode_sentence(data)
    # decoded_sentence = translator.decode_sequence(input_seq)
    # # decoded_sentence = ''
    translate_model = Translator()
    result_text = translate_model.translate(text = data)
    return templates.TemplateResponse('hasil_translate.html', {'request': request, 'data':data, 'result': result_text})

##############################################################################################################
def kataSunda():
    with open(r'kamus_sunda.txt') as word_file:
        return set(word.strip().lower() for word in word_file)  

def levenshtein_distance(str1, str2):
    # Membuat matriks yang terdiri dari baris dan kolom sesuai dengan panjang dari kedua string
    rows = len(str1) + 1
    cols = len(str2) + 1

    # Inisialisasi matriks dengan nilai-nilai default
    distance_matrix = [[0 for col in range(cols)] for row in range(rows)]
    for i in range(1, rows):
        distance_matrix[i][0] = i
    for i in range(1, cols):
        distance_matrix[0][i] = i

    # Loop melalui setiap karakter dari kedua string untuk menghitung jarak Levenshtein
    for col in range(1, cols):
        for row in range(1, rows):
            if str1[row - 1] == str2[col - 1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[row][col] = min(distance_matrix[row - 1][col] + 1,      # Deletion
                                           distance_matrix[row][col - 1] + 1,      # Insertion
                                           distance_matrix[row - 1][col - 1] + cost) # Substitution

    # Mengembalikan nilai di pojok kanan bawah matriks sebagai hasil dari jarak Levenshtein
    return distance_matrix[row][col]

def ini_kata_sunda(word, kata_sunda):
    return word.lower() in kata_sunda

def modulNasal(word):
  kata_ganti = ["p", "b"]
  kata_ganti2 = ["b", "d", "g", "h", "j", "l", "m", "n", "w", "y"]
  kata_ganti3 = ["c", "s"]
  kata_sunda = kataSunda()

  if word.startswith("m"):
    for pref in kata_ganti:
      morfoWord = word.replace("m", pref)
    
    return(morfoWord,ket_ada,'sukses')

  if word.startswith("nga"):
    words = word[len('nga'):]
    for pref in kata_ganti2:
      if words.startswith(pref):
        return(words,ket_ada,'sukses')

  if word.startswith("ng"):
    morfoWord3 = word.replace("ng","k", 1)
    return(morfoWord3,ket_ada,'sukses')

  if word.startswith("ny"):
    for pref in kata_ganti3:
      morfoWord4 = word.replace("ny", pref)

    return(morfoWord4,ket_ada,'sukses')
  
  if word.startswith("n"):
    morfoWord2 = word.replace("n","t", 1)
    return(morfoWord2, ket_ada,'sukses')

  if ini_kata_sunda(word,kata_sunda):
      return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
  else:
      return(word,ket_tada,'gagal : ini kata dasar bahasa sunda')

def hilangkanAwalan(word):
    # awalan = ['ba', 'barang', 'di', 'ka', 'pa', 'pada', 'pang', 'para', 'per', 'pi', 'sa', 'sang', 'si', 'silih', 'sili', 'ti', 'ting', 'pating']
    awalan = ['barang','pating','pada', 'pang', 'para', 'per', 'pi', 'pa','ting','ti','silih', 'sili','sang', 'si', 'sa','di', 'ka', 'ba']
    kata_sunda = kataSunda()
    
    for prefix in awalan:
        if  word.startswith(prefix):
            stemmed_word = word[len(prefix):]
            return(stemmed_word,ket_ada,'sukses')
            # if ini_kata_sunda(stemmed_word,kata_sunda):
            #     return(stemmed_word,ket_ada,'sukses')   

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

# def hilangkanPref(word):
#     prefs = ['ba', 'barang', 'di', 'ka', 'N', 'pa', 'pada', 'pang', 'para', 'per', 'pi' 'sa', 'sang', 'si', 'silih', 'sili', 'ti', 'ting', 'pating']
#     nasal = ['nga', 'ng'] #ganti pakai k
#     nasal2 = ['nge']
#     nasal3 = ['m'] #jika huruf awalan m ganti dengan huruf b atau p bandingkan dengan kamus
#     nasal3 = ['ny'] #jika huruf awalan ny ganti dengan huruf c atau s bandingkan dengan kamus
#     nasal4 = ['n'] #jika huruf awalan n ganti dengan huruf t bandingkan dengan kamus
#     kata_sunda = kataSunda()
    
#     for pre in prefs:
#         if  word.startswith(pre):
#             hapusPref = word[len(pre):]
#             if ini_kata_sunda(hapusPref,kata_sunda):
#                 return(hapusPref,ket_ada,'sukses')
    
#     if ini_kata_sunda(word,kata_sunda):
#         return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
#     else:
#         return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSuf(word):
    sufs = ['keun','ning','ing','an','eun','na']
    kata_sunda = kataSunda()
    
    for suf in sufs:
        if  word.endswith(suf):
            hapusSufs = word[:-len(suf)]
            return(hapusSufs,ket_ada,'sukses')
            
    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanSisipan(word):
    konsonan = 'l'
    konsonan2 = 'al'
    kata_sunda = kataSunda()
    
    if  word.startswith(konsonan) and hapusPre.startswith(konsonan2):
        hapusPre = word[len(konsonan):]
        gabungPre = hapusPre.replace(konsonan2,'l')
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
            return(hapusInf,ket_ada,'sukses')

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanBarung(word):
    akhiran = ['eun','an']
    awalan = ['pika', 'pi','sa']
    kata_sunda = kataSunda()

    for sufix in akhiran:
        for prefix in awalan:
            if word.startswith(prefix) and word.endswith(sufix):
                stemmed = word[len(prefix):-len(sufix)]
                print("stemmed: ",stemmed)
                return(stemmed,ket_ada,'sukses')  

    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')

def hilangkanBareng(word):
    kata_sunda = kataSunda()
    awalan = ['pang','kapi','ka','pa']
    akhiran = "na"
    # konsonan = "dipika"

    for prefix in awalan:
      if word.startswith(prefix) and word.endswith(akhiran):
        stemmed_word = word.replace(prefix, "").replace(akhiran, "")
        return(stemmed_word,ket_ada,'sukses')
  
    if ini_kata_sunda(word,kata_sunda):
        return(word,ket_ada,'gagal : ini kata dasar bahasa sunda')
    else:
        return(word,ket_tada,'gagal : kata tidak dikenal')