import os

import cv2
import mediapipe as mp
import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont, ImageTk

# 1. Configuración de Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Se usa el modelo local que viene junto al script: no se descarga nada.
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")


# Opciones del detector para video en tiempo real
if not os.path.exists(MODELO_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo local en {MODELO_PATH}. "
        "Coloca face_landmarker.task junto a malla.py."
    )

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELO_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,          # Obligatorio para detectar emociones
    output_facial_transformation_matrixes=True
)


# Función adaptada de emociones.py: traduce los blendshapes a un emoji
def obtener_emocion(blendshapes):
    """Devuelve (emoji, nombre) según los blendshapes detectados."""
    if not blendshapes:
        return "😐", "Neutral"

    # Diccionario con los nombres de los gestos y sus puntuaciones (0.0 a 1.0)
    scores = {b.category_name: b.score for b in blendshapes[0]}

    # Umbrales para detectar las emociones
    # Ojo: los nombres reales del modelo son 'mouthSmileLeft/Right',
    # no 'smileLeft/Right' (que no existen y siempre dan 0).
    sonrisa = max(scores.get('mouthSmileLeft', 0),
                  scores.get('mouthSmileRight', 0))
    boca_abierta = scores.get('jawOpen', 0)
    cejas_internas = scores.get('browInnerUp', 0)
    tristeza = max(scores.get('mouthFrownLeft', 0),
                   scores.get('mouthFrownRight', 0))

    # Emocionado va ANTES de Feliz: tiene sonrisa alta, pero además la
    # boca abierta y los ojos bien abiertos (risa / entusiasmo).
    if (sonrisa > 0.5 and boca_abierta > 0.25) or \
            (boca_abierta > 0.4 and
             (scores.get('eyeWideLeft', 0) > 0.4 or
              scores.get('eyeWideRight', 0) > 0.4)):
        return "🤩", "Emocionado"
    elif sonrisa > 0.4:
        return "😀", "Feliz"
    elif boca_abierta > 0.5 and scores.get('mouthPucker', 0) < 0.2:
        return "😲", "Sorprendido"
    elif scores.get('browDownLeft', 0) > 0.4 or scores.get('browDownRight', 0) > 0.4:
        return "😠", "Enojado"
    # Triste: comisuras hacia abajo, reforzado con las cejas arqueadas
    # hacia el centro (browInnerUp), señal típica de tristeza.
    elif tristeza > 0.45 or (tristeza > 0.25 and cejas_internas > 0.3):
        return "😢", "Triste"
    elif scores.get('eyeBlinkLeft', 0) > 0.6 and scores.get('eyeBlinkRight', 0) < 0.3:
        return "😉", "Guiño"
    elif scores.get('eyeBlinkRight', 0) > 0.6 and scores.get('eyeBlinkLeft', 0) < 0.3:
        return "😉", "Guiño"

    return "😐", "Neutral"


# Función de mirada: usa los blendshapes eyeLook* para saber hacia dónde
# apuntan los ojos.
def obtener_mirada(blendshapes):
    """Devuelve (emoji_flecha, nombre, cara) según la dirección de la mirada.

    cara es una tupla (glifo, transformación) para dibujar el icono de
    perfil junto a la flecha:
      - None        -> glifo tal cual
      - "flip"      -> espejo horizontal (perfil mirando a la izquierda)
      - ángulo      -> rotación en grados (positivo = mirando arriba)
    """
    CARA_FRENTE = ("🙂", None)
    CARA_IZQUIERDA = ("🗣️", None)
    CARA_DERECHA = ("🗣️", "flip")
    CARA_ARRIBA = ("🗣️", 30)
    CARA_ABAJO = ("🗣️", -30)

    if not blendshapes:
        return "👀", "Frente", CARA_FRENTE

    scores = {b.category_name: b.score for b in blendshapes[0]}

    arriba = max(scores.get('eyeLookUpLeft', 0),
                 scores.get('eyeLookUpRight', 0))
    abajo = max(scores.get('eyeLookDownLeft', 0),
                scores.get('eyeLookDownRight', 0))
    # Ojos hacia la izquierda de la persona: el ojo izquierdo mira hacia
    # afuera y el derecho hacia adentro (y viceversa para la derecha).
    izquierda = max(scores.get('eyeLookOutLeft', 0),
                    scores.get('eyeLookInRight', 0))
    derecha = max(scores.get('eyeLookInLeft', 0),
                  scores.get('eyeLookOutRight', 0))

    # Gana la dirección con mayor puntuación; si ninguna pasa el umbral,
    # se considera que mira al frente.
    UMBRAL = 0.35
    direccion = max(arriba, abajo, izquierda, derecha)
    if direccion < UMBRAL:
        return "👀", "Frente", CARA_FRENTE
    # Los ejes se presentan invertidos respecto a lo que detecta el modelo
    # (la cámara no está espejada), así que se intercambian emojis y textos:
    # arriba<->abajo e izquierda<->derecha.
    if direccion == arriba:
        return "⬇️", "Abajo", CARA_ABAJO
    if direccion == abajo:
        return "⬆️", "Arriba", CARA_ARRIBA
    if direccion == izquierda:
        return "⬅️", "Derecha", CARA_DERECHA
    return "➡️", "Izquierda", CARA_IZQUIERDA

# Configuración personalizada de los puntos
RADIO_PUNTO = 2              # Aumenta este número para puntos más grandes
COLOR_PUNTO = (255, 120, 0)  # Color en formato BGR (Azul, Verde, Rojo)

ANCHO_VENTANA = 640
ALTO_VENTANA = 480

# Fuente de emojis a color (Noto Color Emoji usa bitmaps de tamaño fijo 109)
RUTA_FUENTE_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
TAMANO_EMOJI = 109

# --- Tema oscuro moderno ---
COLOR_BG = "#1a1b26"        # Fondo de la ventana
COLOR_CARD = "#24283b"      # Fondo de las tarjetas/paneles
COLOR_BORDE = "#3b4261"     # Borde sutil de las tarjetas
COLOR_ACENTO = "#7aa2f7"    # Azul de acento (títulos, detalles)
COLOR_TEXTO = "#c0caf5"     # Texto principal
COLOR_MUTED = "#565f89"     # Texto secundario

COLOR_FONDO_PANEL = (36, 40, 59)  # RGB, igual que COLOR_CARD

# Estilos del botón (planos, con hover)
ESTILO_INICIAR = {"bg": "#2ac3a4", "activebackground": "#1ea084"}
ESTILO_DETENER = {"bg": "#f7768e", "activebackground": "#d9546e"}

# Frames del spinner (caracteres braille) que giran en el botón mientras la
# aplicación está ejecutando la detección de landmarks
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# Familia tipográfica: se toma la primera disponible del sistema
def fuente_disponible(root):
    from tkinter import font as tkfont
    instaladas = set(tkfont.families(root))
    for candidata in ("Inter", "Segoe UI", "Ubuntu", "DejaVu Sans", "Arial"):
        if candidata in instaladas:
            return candidata
    return "TkDefaultFont"


class LandmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Face Mesh - Tkinter")
        self.root.configure(bg=COLOR_BG)
        self.fuente = fuente_disponible(root)

        self.cap = None
        self.landmarks_activos = False
        self._photo = None        # Referencia para que Tkinter no borre la imagen
        self._spinner_job = None  # Timer de la animación del spinner
        self._spinner_idx = 0     # Frame actual del spinner
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Fuente de emojis a color (fuera de la pantalla si no existe)
        self.emoji_font = None
        if os.path.exists(RUTA_FUENTE_EMOJI):
            try:
                self.emoji_font = ImageFont.truetype(
                    RUTA_FUENTE_EMOJI, TAMANO_EMOJI)
            except Exception:
                self.emoji_font = None

        # --- Widgets ---

        # Barra de encabezado: título con detalle de acento
        encabezado = tk.Frame(root, bg=COLOR_BG)
        encabezado.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(
            encabezado, text="◉", bg=COLOR_BG, fg=COLOR_ACENTO,
            font=(self.fuente, 16, "bold"),
        ).pack(side="left")
        tk.Label(
            encabezado, text="Detector de emociones y posicion rostro", bg=COLOR_BG,
            fg=COLOR_TEXTO, font=(self.fuente, 16, "bold"),
        ).pack(side="left")
        tk.Label(
            encabezado, text="MediaPipe", bg=COLOR_BG, fg=COLOR_MUTED,
            font=(self.fuente, 10),
        ).pack(side="right", pady=6)

        # Contenedor principal horizontal: video a la izquierda, panel de
        # emociones a la derecha.
        self.contenedor = tk.Frame(root, bg=COLOR_BG)
        self.contenedor.pack(padx=10, pady=6)

        # Área donde se muestra el video.
        # Se le da tamaño fijo y fondo gris oscuro con mensaje, asi si la
        # cámara falla se ve claramente que no hay señal (no "todo oscuro").
        self.video_frame = tk.Frame(
            self.contenedor, bg=COLOR_BORDE, padx=1, pady=1,
        )
        self.video_frame.pack(side="left")
        self.video_label = tk.Label(
            self.video_frame,
            text="Esperando cámara...",
            width=ANCHO_VENTANA,
            height=ALTO_VENTANA,
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            font=(self.fuente, 14),
        )
        self.video_label.pack()

        # Panel lateral derecho con dos paneles apilados: emoción y mirada
        self.panel_lateral = tk.Frame(self.contenedor, bg=COLOR_BG)
        self.panel_lateral.pack(side="right", padx=(12, 0), fill="y")

        self.widgets_emocion = self._crear_panel(
            self.panel_lateral, "E M O C I Ó N")
        self.widgets_mirada = self._crear_panel(
            self.panel_lateral, "M I R A D A")

        # Referencias de las imágenes de cada panel, para que Tkinter no
        # las borre
        self._emoji_photos = {}

        # Muestra los emojis iniciales (neutral / frente)
        emoji, nombre = obtener_emocion(None)
        self.mostrar_emoji(emoji, nombre, self.widgets_emocion, "_emocion")
        emoji, nombre, cara = obtener_mirada(None)
        self.mostrar_emoji(emoji, nombre, self.widgets_mirada, "_mirada", cara)

        # Botón de iniciar / detener landmarks: plano, ancho completo y con
        # efecto hover
        self.boton = tk.Button(
            root,
            text="▶  Iniciar Landmark",
            command=self.toggle,
            fg="#1a1b26",
            font=(self.fuente, 12, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=12,
            **ESTILO_INICIAR,
        )
        self.boton.pack(fill="x", padx=16, pady=(4, 14))
        self.boton.bind("<Enter>", lambda e: self.boton.config(
            bg=self.boton.cget("activebackground")))
        self.boton.bind("<Leave>", lambda e: self.boton.config(
            bg=(ESTILO_DETENER if self.landmarks_activos
                else ESTILO_INICIAR)["bg"]))

        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        # La cámara arranca sola, sin landmarks
        self.iniciar_camara()

    def toggle(self):
        self.landmarks_activos = not self.landmarks_activos
        if self.landmarks_activos:
            self.boton.config(**ESTILO_DETENER)
            # Arranca la animación del ícono "ejecutando aplicación"
            self._spinner_idx = 0
            self._animar_spinner()
        else:
            self._detener_spinner()
            self.boton.config(text="▶  Iniciar Landmark", **ESTILO_INICIAR)

    def _animar_spinner(self):
        """Rota el ícono del botón mientras la aplicación está en ejecución."""
        if not self.landmarks_activos:
            return
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        self.boton.config(
            text=f"{frame}  Ejecutando aplicación...  (clic para detener)")
        self._spinner_idx += 1
        self._spinner_job = self.root.after(90, self._animar_spinner)

    def _detener_spinner(self):
        """Cancela la animación del spinner al detener la detección."""
        if self._spinner_job is not None:
            self.root.after_cancel(self._spinner_job)
            self._spinner_job = None

    def _crear_panel(self, parent, titulo):
        """Crea una tarjeta con título, emoji y nombre; devuelve sus widgets."""
        panel = tk.Frame(
            parent, width=210, height=300,
            bg=COLOR_CARD, highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )
        panel.pack(fill="both", expand=True, pady=6)
        panel.pack_propagate(False)

        # Franja superior de acento
        tk.Frame(panel, bg=COLOR_ACENTO, height=3).pack(fill="x")

        titulo_label = tk.Label(
            panel, text=titulo,
            bg=COLOR_CARD, fg=COLOR_ACENTO,
            font=(self.fuente, 9, "bold"),
        )
        titulo_label.pack(pady=(12, 0))

        emoji_label = tk.Label(panel, bg=COLOR_CARD)
        emoji_label.pack(expand=True)

        nombre_label = tk.Label(
            panel, bg=COLOR_CARD, fg=COLOR_TEXTO,
            font=(self.fuente, 14, "bold"),
        )
        nombre_label.pack(pady=(0, 14))

        return {"emoji": emoji_label, "nombre": nombre_label}

    def _render_glyph(self, glifo):
        """Renderiza un emoji a color en una imagen RGBA recortada al glifo."""
        medida = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = medida.textbbox((0, 0), glifo, font=self.emoji_font,
                               embedded_color=True)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        imagen = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(imagen)
        draw.text((-bbox[0], -bbox[1]), glifo, font=self.emoji_font,
                  embedded_color=True)
        return imagen

    def mostrar_emoji(self, emoji, nombre, widgets, cache_attr, cara=None):
        """Dibuja el emoji a color con PIL y lo muestra en el panel dado.

        Si se pasa cara=(glifo, transformación), dibuja además el icono de
        perfil arriba con la flecha más pequeña debajo.
        """
        widgets["nombre"].config(text=nombre)

        # Solo re-dibujar la imagen cuando cambia el contenido de este panel
        clave = (emoji, cara)
        if clave == getattr(self, cache_attr, None):
            return
        setattr(self, cache_attr, clave)

        if self.emoji_font is None:
            # Sin fuente de emojis: mostrar solo el nombre
            widgets["emoji"].config(image="", text=nombre,
                                    fg=COLOR_TEXTO,
                                    font=(self.fuente, 28, "bold"))
            return

        if cara is None:
            imagen = Image.new("RGB", (150, 150), COLOR_FONDO_PANEL)
            draw = ImageDraw.Draw(imagen)
            # Centrar el emoji en el cuadro de 150x150 (el glifo mide 109px)
            bbox = draw.textbbox((0, 0), emoji, font=self.emoji_font,
                                 embedded_color=True)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((150 - w) / 2 - bbox[0], (150 - h) / 2 - bbox[1]), emoji,
                      font=self.emoji_font, embedded_color=True)
        else:
            imagen = Image.new("RGB", (150, 150), COLOR_FONDO_PANEL)

            # Icono de perfil: frente, izquierda (espejo), derecha,
            # arriba/abajo (rotando el perfil hacia el ángulo indicado)
            glifo, transform = cara
            cabeza = self._render_glyph(glifo)
            if transform == "flip":
                cabeza = cabeza.transpose(Image.FLIP_LEFT_RIGHT)
            elif transform:
                cabeza = cabeza.rotate(transform, expand=True,
                                       resample=Image.BICUBIC)
            alto_cabeza = 90
            escala = alto_cabeza / cabeza.height
            cabeza = cabeza.resize(
                (max(1, round(cabeza.width * escala)), alto_cabeza),
                Image.LANCZOS)
            imagen.paste(cabeza, ((150 - cabeza.width) // 2, 2), cabeza)

            # Flecha pequeña debajo del perfil
            flecha = self._render_glyph(emoji)
            alto_flecha = 46
            escala = alto_flecha / flecha.height
            flecha = flecha.resize(
                (max(1, round(flecha.width * escala)), alto_flecha),
                Image.LANCZOS)
            imagen.paste(flecha,
                         ((150 - flecha.width) // 2, 150 - alto_flecha - 4),
                         flecha)

        self._emoji_photos[cache_attr] = ImageTk.PhotoImage(image=imagen)
        widgets["emoji"].config(image=self._emoji_photos[cache_attr], text="")

    def iniciar_camara(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("No se pudo abrir la cámara")
            self.cap = None
            self.video_label.config(text="No se pudo abrir la cámara")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_VENTANA)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_VENTANA)
        self.actualizar_frame()

    def detener_camara(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="", text="Cámara detenida")

    def actualizar_frame(self):
        if self.cap is None:
            return

        success, frame = self.cap.read()
        if not success:
            self.detener_camara()
            return

        # Solo detectar y dibujar la malla cuando el botón la activa
        if self.landmarks_activos:
            # Convertir a RGB (MediaPipe lo requiere)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 3. Detectar la malla (Face Mesh)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # 4. Dibujar la malla en el frame
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    for landmark in face_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])

                        # Dibujar el punto con el nuevo tamaño y color
                        cv2.circle(frame, (x, y), RADIO_PUNTO, COLOR_PUNTO, -1)

            # 5. Detectar la emoción y la mirada, mostrando sus emojis
            # en los paneles laterales (la mirada lleva icono de perfil)
            emoji, nombre = obtener_emocion(results.face_blendshapes)
            self.mostrar_emoji(emoji, nombre, self.widgets_emocion, "_emocion")
            emoji, nombre, cara = obtener_mirada(results.face_blendshapes)
            self.mostrar_emoji(emoji, nombre, self.widgets_mirada, "_mirada", cara)

        # Mostrar el frame en la ventana de Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(rgb_frame)
        self._photo = ImageTk.PhotoImage(image=imagen)
        self.video_label.config(image=self._photo, text="")

        # Timer que actualiza el frame (~30 fps), equivalente al QTimer
        self.root.after(30, self.actualizar_frame)

    def cerrar(self):
        self.landmarks_activos = False
        self._detener_spinner()
        self.detener_camara()
        self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LandmarkApp(root)
    root.mainloop()
