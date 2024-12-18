import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
import numpy as np
import pygame
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Filtro de Kalman con implementación de fórmula
class KalmanFilter:
    def __init__(self, process_var, measurement_var):
        self.process_var = process_var  # Q: Varianza del proceso
        self.measurement_var = measurement_var  # R: Varianza de medición
        self.estimate = 0.0  # Estimación inicial
        self.error_cov = 1.0  # Covarianza del error inicial

    def apply(self, data):
        filtered = []
        for value in data:
            # Predicción
            self.error_cov += self.process_var  # Actualización de covarianza de error
            kalman_gain = self.error_cov / (self.error_cov + self.measurement_var)  # Ganancia de Kalman
            self.estimate += kalman_gain * (value - self.estimate)  # Actualización de la estimación
            self.error_cov *= (1 - kalman_gain)  # Actualización de la covarianza de error
            filtered.append(self.estimate)
        return np.array(filtered)

# Función de filtro de Promedio Móvil
def filtro_promedio_movil(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Función de filtro de Mediana
def filtro_median(data, window_size):
    return np.array([np.median(data[max(i - window_size // 2, 0): min(i + window_size // 2, len(data))]) for i in range(len(data))])

# Variables globales
audio = audio_ruido = audio_filtrado = None
escala_ruido = escala_proc = escala_meas = None
filtro_seleccionado = None

# Función para graficar audio
def graficar_audio(audio_graficar, ax, titulo):
    muestras = np.array(audio_graficar.get_array_of_samples())
    tiempo = np.linspace(0, len(muestras) / audio_graficar.frame_rate, num=len(muestras))

    ax.plot(tiempo, muestras, color="blue")
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")

# Ventana 1: Cargar audio
def ventana_cargar_audio():
    def cargar_audio():
        global audio
        ruta = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.mp3;*.wav")])
        if ruta:
            audio = AudioSegment.from_file(ruta)
            messagebox.showinfo("Éxito", "Audio cargado correctamente.")
            ventana.destroy()
            ventana_grafica_original()

    ventana = tk.Toplevel(root)
    ventana.title("Cargar Audio")
    ventana.geometry("800x600")

    tk.Label(ventana, text="Paso 1: Cargar un archivo de audio").pack(pady=10)
    tk.Button(ventana, text="Cargar Audio", command=cargar_audio).pack(pady=10)
    tk.Button(ventana, text="Siguiente", command=lambda: ventana_grafica_original()).pack(pady=20)

# Ventana 2: Mostrar gráfica del audio original y agregar ruido
def ventana_grafica_original():
    def agregar_ruido():
        global audio_ruido
        tipo = "blanco"
        ruido = np.random.normal(0, 1, int(audio.frame_rate * len(audio) / 1000))
        ruido = (ruido * 32767 * escala_ruido.get() / 100).astype(np.int16)
        audio_ruido = audio.overlay(AudioSegment(ruido.tobytes(), frame_rate=audio.frame_rate, sample_width=2, channels=1))
        messagebox.showinfo("Éxito", "Ruido agregado correctamente.")

    ventana = tk.Toplevel(root)
    ventana.title("Gráfica del Audio Original y Agregar Ruido")
    ventana.geometry("800x600")

    tk.Label(ventana, text="Paso 2: Gráfica del Audio Original").pack(pady=10)

    # Frame para la gráfica del audio original y el botón
    frame_original = tk.Frame(ventana)
    frame_original.pack(pady=20)

    fig, ax = plt.subplots(figsize=(5, 3))  # Tamaño reducido de la gráfica
    graficar_audio(audio, ax, "Audio Original")
    canvas = FigureCanvasTkAgg(fig, master=frame_original)
    canvas.get_tk_widget().pack()
    canvas.draw()

    escala_ruido = tk.Scale(ventana, from_=0, to=100, label="Intensidad Ruido", orient="horizontal")
    escala_ruido.pack(pady=10)

    tk.Button(frame_original, text="Agregar Ruido", command=agregar_ruido).pack(pady=10)
    tk.Button(ventana, text="Siguiente", command=ventana_tipo_filtro).pack(pady=20)

# Ventana 3: Elección del filtro y mostrar fórmula
def ventana_tipo_filtro():
    def elegir_filtro():
        global filtro_seleccionado
        filtro_seleccionado = filtro_var.get()
        ventana.destroy()
        ventana_aplicar_filtro()

    ventana = tk.Toplevel(root)
    ventana.title("Seleccionar Tipo de Filtro")
    ventana.geometry("800x600")

    tk.Label(ventana, text="Paso 3: Seleccionar el tipo de filtro a aplicar").pack(pady=10)

    filtro_var = tk.StringVar()

    # Radio buttons para elegir el filtro
    tk.Radiobutton(ventana, text="Filtro Kalman", variable=filtro_var, value="Kalman").pack(pady=5)
    tk.Radiobutton(ventana, text="Filtro Promedio Móvil", variable=filtro_var, value="Promedio Móvil").pack(pady=5)
    tk.Radiobutton(ventana, text="Filtro Mediana", variable=filtro_var, value="Mediana").pack(pady=5)

    tk.Button(ventana, text="Seleccionar Filtro", command=elegir_filtro).pack(pady=20)

# Ventana 4: Gráficas y aplicar filtro
def ventana_aplicar_filtro():
    def aplicar_filtro():
        global audio_filtrado
        muestras = np.array(audio_ruido.get_array_of_samples())
        
        if filtro_seleccionado == "Kalman":
            kalman = KalmanFilter(escala_proc.get() / 1000, escala_meas.get() / 1000)
            filtrado = kalman.apply(muestras).astype(np.int16)
        elif filtro_seleccionado == "Promedio Móvil":
            window_size = escala_proc.get()
            filtrado = filtro_promedio_movil(muestras, window_size=window_size).astype(np.int16)
        elif filtro_seleccionado == "Mediana":
            window_size = escala_proc.get()
            filtrado = filtro_median(muestras, window_size=window_size).astype(np.int16)

        audio_filtrado = AudioSegment(filtrado.tobytes(), frame_rate=audio.frame_rate, sample_width=2, channels=1)
        
        messagebox.showinfo("Éxito", f"Filtro {filtro_seleccionado} aplicado.")
        ventana.destroy()
        ventana_grafica_filtrada()

    ventana = tk.Toplevel(root)
    ventana.title("Aplicar Filtro")
    ventana.geometry("800x600")

    tk.Label(ventana, text="Paso 4: Aplicar el Filtro").pack(pady=10)

    # Mostrar la fórmula del filtro seleccionado
    if filtro_seleccionado == "Kalman":
        formula = "Kalman: Estimación = Estimación + (Ganancia de Kalman * (Valor medido - Estimación))"
    elif filtro_seleccionado == "Promedio Móvil":
        formula = "Promedio Móvil: Valor filtrado = Promedio de los valores vecinos en la ventana."
    elif filtro_seleccionado == "Mediana":
        formula = "Mediana: Valor filtrado = Mediana de los valores en la ventana."
    else:
        formula = "Selección de filtro no válida"

    label_formula = tk.Label(ventana, text=formula)
    label_formula.pack(pady=20)

    # Escalas de varianza, ahora en Ventana 4
    global escala_proc, escala_meas

    escala_proc = tk.Scale(ventana, from_=1, to=100, label="Tamaño Ventana o Varianza", orient="horizontal")
    escala_proc.pack(pady=10)

    # Agregar la escala de medición (escala_meas) solo para Kalman
    escala_meas = tk.Scale(ventana, from_=1, to=100, label="Varianza de Medición", orient="horizontal")
    escala_meas.pack(pady=10)

    tk.Button(ventana, text="Aplicar Filtro", command=aplicar_filtro).pack(pady=20)

# Ventana 5: Gráficas del audio original, con ruido y filtrado con botones de reproducción y guardar
def ventana_grafica_filtrada():
    ventana = tk.Toplevel(root)
    ventana.title("Audio Filtrado")
    ventana.geometry("800x600")

    tk.Label(ventana, text="Paso 5: Gráficas de los Audios").pack(pady=10)

    # Crear una figura con 3 subgráficas en una fila horizontal
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Tamaño reducido de la figura
    fig.subplots_adjust(hspace=0.5)  # Añadir espacio entre las subgráficas

    # Graficar los tres audios
    graficar_audio(audio, axs[0], "Audio Original")
    graficar_audio(audio_ruido, axs[1], "Audio con Ruido")
    graficar_audio(audio_filtrado, axs[2], "Audio Filtrado")

    # Mostrar la figura con las tres gráficas
    canvas = FigureCanvasTkAgg(fig, master=ventana)
    canvas.get_tk_widget().pack()
    canvas.draw()

    # Frame para los botones de reproducción debajo de cada gráfica
    frame_botones = tk.Frame(ventana)
    frame_botones.pack(side="bottom", pady=10)

    # Reproducir audio original
    frame_original = tk.Frame(frame_botones)
    frame_original.pack(pady=5)
    tk.Button(frame_original, text="Reproducir Original", command=lambda: reproducir(audio)).pack()

    # Reproducir audio con ruido
    frame_ruido = tk.Frame(frame_botones)
    frame_ruido.pack(pady=5)
    tk.Button(frame_ruido, text="Reproducir con Ruido", command=lambda: reproducir(audio_ruido)).pack()

    # Reproducir audio filtrado
    frame_filtrado = tk.Frame(frame_botones)
    frame_filtrado.pack(pady=5)
    tk.Button(frame_filtrado, text="Reproducir Filtrado", command=lambda: reproducir(audio_filtrado)).pack()

    # Botón para guardar el audio filtrado
    tk.Button(frame_botones, text="Guardar Audio", command=guardar_audio).pack(pady=10)

# Función para reproducir el audio
def reproducir(audio):
    if audio:
        pygame.mixer.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            ruta_temporal = temp_file.name
            audio.export(ruta_temporal, format="wav")
        pygame.mixer.music.load(ruta_temporal)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            root.update()

# Función para guardar el audio filtrado
def guardar_audio():
    if audio_filtrado:
        ruta = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Archivos WAV", "*.wav")])
        if ruta:
            audio_filtrado.export(ruta, format="wav")
            messagebox.showinfo("Éxito", "Audio guardado correctamente.")

# Ventana principal
root = tk.Tk()
root.title("Procesador de Audio con Filtro")
root.geometry("300x200")

tk.Label(root, text="Procesador de Audio con Filtro").pack(pady=20)
tk.Button(root, text="Iniciar", command=ventana_cargar_audio).pack()

root.mainloop()
