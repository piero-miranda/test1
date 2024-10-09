import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Cargar el modelo de TensorFlow
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('SegNet_trained.h5')  # Cambia la ruta si es necesario
    return model

model = load_model()

def main():
    st.title("Sistema de Gestión Médica")
    st.sidebar.title("Menú de Navegación")

    # Menú de opciones
    menu = ["Inicio", "Registrarse", "Iniciar Sesión", "Panel del Doctor"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Página de inicio
    if choice == "Inicio":
        st.subheader("Página de Inicio")
        st.write("Bienvenido al Sistema de Gestión Médica. Aquí puedes gestionar pacientes y analizar imágenes médicas.")

    # Registro de nuevos usuarios
    elif choice == "Registrarse":
        st.subheader("Crear una nueva cuenta")
        new_username = st.text_input("Nombre de usuario")
        new_password = st.text_input("Contraseña", type='password')
        confirm_password = st.text_input("Confirmar Contraseña", type='password')
        if st.button("Registrarse"):
            if new_password == confirm_password:
                st.success("Registrado con éxito")
            else:
                st.error("Las contraseñas no coinciden")

    # Inicio de sesión
    elif choice == "Iniciar Sesión":
        st.subheader("Iniciar sesión")
        username = st.text_input("Nombre de usuario")
        password = st.text_input("Contraseña", type='password')
        if st.button("Login"):
            st.success(f"Has iniciado sesión como {username}")

    # Panel del Doctor para gestionar pacientes y cargar imágenes
    elif choice == "Panel del Doctor":
        st.subheader("Panel del Doctor")
        st.info("Gestiona los pacientes y analiza imágenes médicas.")

        # Selección entre registrar o buscar pacientes
        patient_option = st.selectbox("Seleccionar una opción", ["Elegir paciente", "Registrar paciente nuevo"])
        
        if patient_option == "Registrar paciente nuevo":
            st.subheader("Registrar nuevo paciente")
            p_name = st.text_input("Nombre completo")
            p_age = st.number_input("Edad", min_value=1, max_value=100)
            p_sex = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
            p_id = st.text_input("Número de DNI")
            if st.button("Guardar perfil"):
                st.success(f"Perfil de {p_name} guardado correctamente.")
        
        elif patient_option == "Elegir paciente":
            st.subheader("Buscar paciente existente")
            search = st.text_input("Buscar paciente por DNI")
            if st.button("Buscar"):
                # Simulación de búsqueda (agrega la lógica real si la tienes)
                st.write(f"Resultados de la búsqueda para el DNI: {search}")
        
        # Subida de imagen para análisis médico
        st.subheader("Análisis de imágenes médicas")
        image_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption='Imagen cargada correctamente', use_column_width=True)
            
            if st.button('Procesar imagen'):
                processed_image = process_image(image, model)  # Función para procesar la imagen
                st.image(processed_image, caption='Máscara segmentada', use_column_width=True)

# Función para preprocesar y procesar la imagen utilizando el modelo
def process_image(image, model):
    try:
        # Preprocesar la imagen
        image = np.array(image)
        image = cv2.resize(image, (224, 224))  # Cambiar a las dimensiones necesarias
        image = image / 255.0  # Normalizar la imagen
        image = np.expand_dims(image, axis=0)

        # Hacer la predicción con el modelo
        prediction = model.predict(image)

        # Procesar la predicción para hacerla visible (normalizar a 0-255)
        mask = prediction.squeeze()  # Eliminar dimensiones adicionales
        mask = (mask * 255).astype(np.uint8)  # Normalizar a rango [0, 255]

        return mask  # Retornar la máscara segmentada como imagen
    except Exception as e:
        st.write(f"Error al procesar la imagen: {e}")
        return None

# Ejecutar la aplicación principal
if __name__ == '__main__':
    main()
