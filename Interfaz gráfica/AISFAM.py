import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import csv

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def generate_X():
    X = [0] * 32  # Inicializa el vector de 1x32 con ceros
    index = 0  # Índice para recorrer X
    
    # Pregunta 1 (Single: SI/NO)
    X[index] = 1 if responses[0] and responses[0].get() == "SI" else 0
    index += 1
    
    # Pregunta 2 (Multiple: 3 opciones)
    for i in range(3):
        X[index] = responses[1][i].get() if responses[1] else 0
        index += 1
    
    # Pregunta 3 (Multiple: 4 opciones)
    for i in range(4):
        X[index] = responses[2][i].get() if responses[2] else 0
        index += 1
    
    # Pregunta 4 (Multiple: 3 opciones)
    for i in range(3):
        X[index] = responses[3][i].get() if responses[3] else 0
        index += 1
    
    # Pregunta 5 (Multiple: 4 opciones)
    for i in range(4):
        X[index] = responses[4][i].get() if responses[4] else 0
        index += 1
    
    # Pregunta 6 (Single: SI/NO)
    X[index] = 1 if responses[5] and responses[5].get() == "SI" else 0
    index += 1
    
    # Pregunta 7 (Single: SI/NO)
    X[index] = 1 if responses[6] and responses[6].get() == "SI" else 0
    index += 1
    
    # Pregunta 8 (Multiple: 2 opciones)
    for i in range(2):
        X[index] = responses[7][i].get() if responses[7] else 0
        index += 1
    
    # Pregunta 9 (Multiple: 2 opciones)
    for i in range(2):
        X[index] = responses[8][i].get() if responses[8] else 0
        index += 1
    
    # Pregunta 10 (Single: SI/NO)
    X[index] = 1 if responses[9] and responses[9].get() == "SI" else 0
    index += 1
    
    # Pregunta 11 (Multiple: 4 opciones)
    for i in range(4):
        X[index] = responses[10][i].get() if responses[10] else 0
        index += 1
    
    # Pregunta 12 (Single: SI/NO)
    X[index] = 1 if responses[11] and responses[11].get() == "SI" else 0
    index += 1
    
    # Pregunta 13 (Single: SI/NO)
    X[index] = 1 if responses[12] and responses[12].get() == "SI" else 0
    index += 1
    
    # Pregunta 14 (Single: SI/NO)
    X[index] = 1 if responses[13] and responses[13].get() == "SI" else 0
    index += 1
    
    # Pregunta 15 (Multiple: 2 opciones)
    for i in range(2):
        X[index] = responses[14][i].get() if responses[14] else 0
        index += 1
    
    # Pregunta 16 (Single: SI/NO)
    X[index] = 1 if responses[15] and responses[15].get() == "SI" else 0

    return X

# Preguntas y opciones
questions = [
    {"text": "¿El eje está fracturado en dos o más piezas separadas?", "options": ["SI", "NO"], "type": "single"},
    {"text": "Identifique el o los patrones de marcas observadas en la superficie de fractura.", "options": ["Marcas de playa", "Marcas radiales", "Marcas ratchet"], "type": "multiple"},
    {"text": "Identifique la o las apariencias en la superficie de fractura.", "options": ["Apariencia granular", "Apariencia tersa", "Apariencia fibrosa o desgarre", "Dos zonas en superficie de fractura con diferentes texturas"], "type": "multiple"},
    {"text": "Seleccione el tipo o los tipos de fractura presentada en superficie de la pieza analizada.", "options": ["Fractura transversal", "Fractura diagonal a 45° o inicio longitudinal", "Fractura transversal con grietas radiales a 45°"], "type": "multiple"},
    {"text": "Identifique el o los patrones de deformación en la superficie analizada.", "options": ["Deformación plástica en dirección de la rotación o entorchamiento", "Sin deformación plástica", "Deformación plástica a flexión o doblado", "Deformación en estrías o chaveteros o adelgazamiento de dientes"], "type": "multiple"},
    {"text": "¿Observa residuos de corrosión en el eje?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿Durante su operación, el ambiente circundante al eje era corrosivo?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿El torque que soportaba el eje durante su operación?", "options": ["Torque constante", "Torque variable"], "type": "multiple"},
    {"text": "¿Geométricamente de la zona analizada es?", "options": ["Sólida o tubular de pared gruesa", "Tubular de pared delgada"], "type": "multiple"},
    {"text": "¿Se presenta modificación superficial en el eje?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿Encuentra alguno de los siguientes en las zonas de contacto del eje analizado?", "options": ["Picaduras", "Grietas superficiales", "Rayones en la superficie", "Aspecto abrillantado"], "type": "multiple"},
    {"text": "¿Observa transferencia de mental o arrastre en el eje?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿Observa evidencia de calor o fusión en el eje?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿Observa residuos de corrosión, depósitos o cambios de color?", "options": ["SI", "NO"], "type": "single"},
    {"text": "¿Qué tipo de daño observa en la superficie?", "options": ["Daño localizado", "Daño homogéneo"], "type": "multiple"},
    {"text": "¿La zona del eje analizada tiene ajuste prieto?", "options": ["SI", "NO"], "type": "single"}
]

# Almacenar respuestas
responses = [None] * len(questions)

# Crear ventana principal
root = tk.Tk()
root.title("Sistema experto para diagnóstico de modos de falla en ejes")
root.geometry("809x500")

main_frame = tk.Frame(root)
#main_frame.pack(padx=10, pady=100)
main_frame.pack()
# Common text widget size
text_width = 20
text_height = 24

current_question = 0

# Widgets
question_label = tk.Label(root, text="", wraplength=600, font=("Cambria", 10))
question_label.pack(pady=20)

options_frame = tk.Frame(root)
options_frame.pack()

navigation_frame = tk.Frame(root)
navigation_frame.pack(pady=20)

prev_button = tk.Button(navigation_frame, text="Anterior", command=lambda: navigate(-1))
next_button = tk.Button(navigation_frame, text="Siguiente", command=lambda: navigate(1))
check_button = tk.Button(root, text="Resumen Cuestionario", command=lambda: check_answers())
validar_button = tk.Button(root, text="Validar Modelos", command=lambda: validarModelos2())
#validar_button = tk.Button(root, text="Validar modelos", command = lambda: checkModelo())

prev_button.pack(side=tk.LEFT, padx=10)
next_button.pack(side=tk.RIGHT, padx=10)

# Funciones
def reset_responses_from_question_2():
    global responses
    for i in range(1, len(responses)):
        responses[i] = None

def get_question_sequence():
    if responses[0] is not None and responses[0].get() == "SI":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        return [0, 4, 8, 9, 10, 11, 12, 13, 14, 15]

def display_question(index):
    global current_question
    current_question = index

    question_label.config(text=questions[index]["text"])

    for widget in options_frame.winfo_children():
        widget.destroy()

    if questions[index]["type"] == "single":
        if responses[index] is None:
            responses[index] = tk.StringVar(value="")
        combo = ttk.Combobox(options_frame, textvariable=responses[index], state="readonly")
        combo['values'] = ("SI", "NO")
        combo.pack()

        if index == 0:
            combo.bind("<<ComboboxSelected>>", lambda e: reset_responses_from_question_2())

    elif questions[index]["type"] == "multiple":
        if responses[index] is None:
            responses[index] = [tk.IntVar(value=0) for _ in questions[index]["options"]]
        for i, option in enumerate(questions[index]["options"]):
            tk.Checkbutton(options_frame, text=option, variable=responses[index][i]).pack(anchor=tk.W)

    prev_button.pack_forget() if index == 0 else prev_button.pack(side=tk.LEFT, padx=10)
    next_button.pack_forget() if index == len(questions) - 1 else next_button.pack(side=tk.RIGHT, padx=10)
    if index == 8 or index == 15:
        check_button.pack(side=tk.RIGHT, padx=10)

        #validar_button = tk.Button(root, text="Validar Modelos", command=validarModelos)
        #validar_button.pack(pady=11)
        validar_button.pack(side=tk.RIGHT, padx=11)
    else:
        check_button.pack_forget()

def navigate(step):
    sequence = get_question_sequence()
    new_index = sequence.index(current_question) + step if current_question in sequence else 0
    if 0 <= new_index < len(sequence):
        display_question(sequence[new_index])

def save_answers():
    with open('respuestas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pregunta", "Respuesta"])
        for index, response in enumerate(responses):
            if isinstance(response, tk.StringVar):
                writer.writerow([questions[index]['text'], response.get()])
            elif isinstance(response, list):
                selected_options = [questions[index]['options'][i] for i, var in enumerate(response) if var.get() == 1]
                writer.writerow([questions[index]['text'], ', '.join(selected_options)])

    messagebox.showinfo("Guardado exitoso", "Las respuestas se han guardado satisfactoriamente.")        # Mostrar aviso de éxito

def check_answers():
    top = tk.Toplevel(root)
    top.title("Comprobación de síntomas")
    top.geometry("600x400")

    save_button = tk.Button(top, text="Guardar respuestas", command=save_answers)
    save_button.pack(pady=10)
  
    text = tk.Text(top, wrap=tk.WORD)
    text.pack(expand=True, fill=tk.BOTH)
    for index, response in enumerate(responses):
        if isinstance(response, tk.StringVar):
            text.insert(tk.END, f"{questions[index]['text']} --- {response.get()}\n")
        elif isinstance(response, list):
            selected_options = [questions[index]['options'][i] for i, var in enumerate(response) if var.get() == 1]
            text.insert(tk.END, f"{questions[index]['text']} .... {', '.join(selected_options)}\n")

    X = generate_X()
    vector_text = "[" + ",".join(map(str, X)) + "]"  # Convierte X en una cadena con números separados por comas
    text.insert(tk.END, vector_text)
    with open('vector_x.txt', mode='w', encoding='utf-8') as text_file:     # Guardar exactamente lo que se muestra en pantalla en un archivo .txt
        text_file.write(vector_text)
        
def validarModelos():
    top = tk.Toplevel(root)
    top.title("Comprobación de síntomas")
    top.geometry("300x400")

# =============================================================================
# Cargar modelo y predecir con nuevo vector
# =============================================================================
def predecir_modo_falla(entrada_procesada, modelo_cargado):
    """
    Recibe un vector numpy de forma (1, 32) con valores binarios (0/1).
    Devuelve las probabilidades para cada modo de falla.
    """
    # Predecir probabilidades
    probabilidades = modelo_cargado.predict(entrada_procesada, verbose=0)
    
    # Formatear resultados
    #clases = encoder.categories_[0]
    clases = ['abw','adw', 'bcf', 'bfb', 'bff', 'bpf', 'dkws', 'fddf', 'fdff', 'lu', 'pc', 'sbub', 'sbut', 'sccub', 'sf', 'tbf', 'tcf', 'tdf', 'tff', 'tffss', 'tpf', 'tscc', 'twdf', 'ur']
    resultados = {clase: prob for clase, prob in zip(clases, probabilidades[0])}
    
    return resultados
# =============================================================================

def create_widget(parent, text, column):
    frame = tk.Frame(parent)
    frame.grid(row=10, column=column, padx=10, pady=10)

    label = tk.Label(frame, text=text)
    label.pack()

    text_widget = tk.Text(frame, width=text_width, height=text_height, wrap=tk.WORD)
    text_widget.pack()

    return text_widget
# =============================================================================
def uniquePath(path):
    '''
    # Esta función verifica que exista sólo un archivo con ese nombre en la ruta, si existe entonces agrega un contador al nombre del archivo para que luego se cree
    # un nuevo archivo con ese nombre nuevo

    '''
    fileName, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = fileName+"_"+str(counter)+extension
        counter +=1
    return path
# =============================================================================
def validarModelos2():

    with open('vector_x.txt') as f:
        lines = [line.rstrip() for line in f]
    
    sourcePath = r"C:\Users\mario.adame\.spyder-py3\SECMF\sistema_experto" 
    outputPath = r"C:\Users\mario.adame\.spyder-py3\SECMF\sistema_experto" 

    modosFallalst = ("mlp_bm_gkv","dnn_bm_gkv","lstm_bm_gkv")
    for line in lines:
        prueba = line
        print(prueba)
        break
    prueba = prueba.replace('[','')
    prueba = prueba.replace(']','')
    prueba = prueba.split(',')
    newstr = [int(item) for item in prueba]

    pruebaMod = np.array(newstr)

    for falla in modosFallalst:
        print(falla)
        #modelo_modos_falla
        modelo_cargado = tf.keras.models.load_model(sourcePath +'\\' + falla + '.keras')
        if (falla == "lstm_bm_gkv"):
            entrada= pruebaMod.reshape(1,1,32)
        else:
            entrada= pruebaMod.reshape(1,32)
        # calcular las probabilidades
        probabilidades = predecir_modo_falla(entrada, modelo_cargado)
        if (falla == "mlp_bm_gkv"):
            text = create_widget(main_frame, "Inferencia MLP", 0)
        elif (falla == "dnn_bm_gkv"):
            text = create_widget(main_frame, "Inferencia DNN", 1)
        elif (falla == "lstm_bm_gkv"):
            text = create_widget(main_frame, "Inferencia LSTM", 2)

        # Obtener ruta del archivo de salida
        outputFileName = falla + ".txt"
        outputFile = uniquePath(outputPath+"/"+outputFileName)

        # Imprimir resultados ordenados de mayor a menor probabilidad
        output_file = open(outputFile,"x")
        output_file.write("\nProbabilidades por modo de falla:")    
        for modo, prob in sorted(probabilidades.items(), key=lambda x: x[1], reverse=True):
            output_file.write(f"{modo}: {prob*100:.2f}%\n")
            text.insert(tk.END, f"{modo} --- {prob*100:.2f}\n")
        output_file.close()
        #-----
    #-----

display_question(0)
root.mainloop()