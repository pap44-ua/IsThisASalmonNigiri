# 🍣 Detector de Nigiri de Salmón - Guía Completa

## ¿Qué es este proyecto?

Este proyecto utiliza **Inteligencia Artificial** para reconocer si una foto es un **nigiri de salmón** o **no lo es**. Es como enseñarle a una máquina a identificar comida mostrándole muchos ejemplos, hasta que aprenda a diferenciarlas por sí sola.

---

## 📚 Conceptos básicos (Sin tecnicismos)

### 1. **¿Qué es una Red Neuronal?**

Imagina que alguien te muestra fotos de nigiri de salmón y otras comidas, pero no te dice cuál es cuál. Después de ver cientos de fotos, tu cerebro aprende los patrones:
- El nigiri tiene forma rectangular
- Tiene arroz blanco/beige
- Encima tiene salmón de color naranja/rojo
- Está sobre una base oscura

Una **Red Neuronal Artificial** es un programa que funciona igual: ve muchas imágenes y aprende a reconocer patrones automáticamente.

### 2. **¿Cómo aprende?**

El proceso es similar a aprender a cocinar:
1. **Ves un ejemplo** (una foto de nigiri)
2. **Haces una predicción** ("Creo que es un nigiri")
3. **Compruebas si acertaste** (Alguien te dice "sí, acertaste" o "no, fallaste")
4. **Ajustas tu estrategia** (Recuerdas los detalles que te ayudaron a acertar)
5. **Repites** muchas veces hasta ser experto

Esto es exactamente lo que hace la IA: **entrenar**.

### 3. **Tipos de Imágenes que Maneja**

- ✅ Nigiri de salmón (la que queremos detectar)
- ❌ Otras cosas (cualquier cosa que NO sea nigiri de salmón)

---

## 🏗️ Estructura del Proyecto

```
IsThisASalmonNigiri/
├── main.py                 ← Aquí se entrena el modelo
├── app.py                  ← Aquí está la interfaz web
├── requirements.txt        ← Librerías necesarias
├── modelo_nigiri.h5        ← El modelo entrenado (se crea al ejecutar main.py)
└── dataset/
    ├── nigiri_salmon/      ← Fotos de nigiri de salmón
    └── no_nigiri/          ← Fotos de otras cosas
```

---

## 📁 Las Carpetas de Datos (Dataset)

### ¿Por qué necesitamos dos carpetas?

**Carpeta 1: `nigiri_salmon/`**
- Contiene fotos de nigiri de salmón
- Cuantas más, mejor (100+ imágenes ideales)
- Diferentes ángulos, iluminaciones, etc.

**Carpeta 2: `no_nigiri/`**
- Contiene fotos de CUALQUIER OTRA COSA
- Puede ser: pizza, sushi diferente, manzanas, coches, etc.
- También necesita muchas fotos (100+ ideales)

**¿Por qué dos categorías?**
La máquina aprende por contraste. Si solo ves nigiri, no sabes qué NO es nigiri. Es como si alguien solo te mostrara gatos y luego le preguntaras "¿es esto un gato o un perro?". Si nunca viste un perro, no podrías decirlo.

---

## 🧠 El Modelo de IA Explicado

### ¿Qué sucede cuando ejecutas `python main.py`?

#### **PASO 1: Cargar las imágenes**
```
📂 dataset/
   ├── nigiri_salmon/ → Lee todas las fotos aquí
   └── no_nigiri/ → Lee todas las fotos aquí
```

El programa:
1. Abre cada foto
2. La convierte a tamaño estándar (128x128 píxeles)
3. La convierte a números (cada píxel tiene valores de color)

**Analogía:** Como si escanearas las fotos para convertirlas en datos que el ordenador pueda entender.

#### **PASO 2: Preparar los datos**

**Normalización:**
- Cada píxel tiene valores de 0-255 (negro a blanco)
- Se convierten a 0-1 (más fácil para la máquina)

**División de datos:**
- 80% para **entrenar** (aprender)
- 20% para **probar** (ver si realmente aprendió)

**Analogía:** Como estudiar con 80 ejercicios y hacer un examen con 20 preguntas.

#### **PASO 3: Data Augmentation (Augmentación de Datos)**

Esto es un **truco muy importante**: si tienes 100 fotos, artificialmente se pueden crear 1000+ variaciones:

- 🔄 **Rotación:** Gira la foto 20 grados
- ↔️ **Desplazamiento:** Mueve la foto hacia los lados
- 🔍 **Zoom:** Amplía o reduce
- 💡 **Brillo:** Hace más clara u oscura
- 🔀 **Espejo:** Voltea horizontalmente

**¿Por qué?**
Así la máquina aprende que un nigiri sigue siendo nigiri aunque esté:
- Rotado
- Iluminado de forma diferente
- Fotografiado desde otro ángulo

**Analogía:** Es como si el profesor te mostrara la misma pregunta de examen con diferentes palabras para asegurase de que realmente entiendes, no solo que memorizaste.

#### **PASO 4: La Red Neuronal (La Arquitectura)**

Nuestra red tiene esta estructura:

```
ENTRADA (Imagen 128×128 píxeles)
    ↓
[Capa 1] Conv2D - 64 filtros
    ↓ Aprende patrones simples (líneas, esquinas)
[Capa 2] MaxPooling - Comprime la información
    ↓ Reduce el tamaño a la mitad
[Capa 3] Conv2D - 128 filtros
    ↓ Aprende patrones más complejos
[Capa 4] MaxPooling
    ↓ Comprime más
[Capa 5] Conv2D - 256 filtros
    ↓ Aprende patrones muy complejos
[Capa 6] MaxPooling
    ↓ Comprime más
[Capa 7] Flatten - Convierte en lista única
    ↓ 
[Capa 8] Dense - 256 neuronas
    ↓ Analiza toda la información
[Capa 9] Dropout - Evita memorizar
    ↓
[Capa 10] Dense - 128 neuronas
    ↓
[SALIDA] 2 resultados: "Es nigiri" o "No es nigiri"
```

**¿Cómo funciona cada parte?**

**Conv2D (Convolución):**
- Busca **características** en la imagen
- Primera capa: encuentra líneas, bordes, esquinas
- Segunda capa: combina eso en formas simples (círculos, rectángulos)
- Tercera capa: reconoce objetos (arroz, salmón, plato)

**Analogía:** Es como si entrecieras los ojos para ver solo sombras, luego abres un poco más para ver formas, luego completamente para ver detalles.

**MaxPooling:**
- Reduce el tamaño de la información
- Mantiene lo más importante
- Analogía: Pasar de tomar notas de todo a solo lo esencial

**Dropout:**
- Desactiva aleatoriamente neuronas
- Evita que la máquina "memorice" en lugar de "aprender"
- Analogía: Como estudiar a veces sin tus apuntes para aseguarte de que realmente entiendes.

**Dense (Capas totalmente conectadas):**
- Unen toda la información
- Toman la decisión final
- Analogía: Un jurado que decide basándose en toda la evidencia

#### **PASO 5: Entrenamiento**

```
Para cada época (50 veces):
    Para cada imagen de entrenamiento:
        1. La red ADIVINA si es nigiri o no
        2. Comprueba si acertó o no
        3. Si se equivocó, ajusta sus "pesos"
           (números internos que definen qué es importante)
        4. Repite con la siguiente imagen
    
    Después de ver todas las imágenes:
        5. Prueba con las imágenes de prueba
        6. Si empieza a empeorar (overfitting), se detiene
```

**Early Stopping:**
- Si después de 5 épocas no mejora, se detiene
- Evita desperdiciar tiempo y que "memorice"

**¿Cuántas épocas?**
Una **época** es ver todas las imágenes de entrenamiento una vez. Con 50 épocas, ve cada imagen 50 veces.

**Analogía:** Como estudiar un tema 50 veces hasta que lo domines.

#### **PASO 6: Evaluación**

Después del entrenamiento:
- Se prueban todas las imágenes de prueba
- Se calcula la **precisión** (% de aciertos)
- Se guarda el modelo como `modelo_nigiri.h5`

---

## 🎯 La Interfaz Web (app.py)

### ¿Qué hace `streamlit run app.py`?

Crea una **página web** en tu navegador donde puedes:

1. **Subir una imagen** desde tu ordenador
2. **Ver una vista previa** de la foto
3. **Hacer clic en "Analizar"**
4. **Recibir el resultado:**
   - ✅ "Es un nigiri de salmón" (con % de confianza)
   - ❌ "NO es un nigiri de salmón" (con % de confianza)

### ¿Qué sucede internamente?

```
Subes una foto
    ↓
La aplicación la carga
    ↓
La redimensiona a 128x128 (mismo tamaño que el entrenamiento)
    ↓
La normaliza (valores 0-1)
    ↓
La pasa al modelo guardado (modelo_nigiri.h5)
    ↓
El modelo calcula probabilidades:
    - Probabilidad de ser nigiri: 85%
    - Probabilidad de no ser nigiri: 15%
    ↓
Muestra el resultado con la confianza más alta (85%)
```

---

## 🔄 El Flujo Completo

### Primera vez (Entrenar el modelo):

```
1. Organiza tus fotos:
   dataset/
   ├── nigiri_salmon/ (100+ fotos)
   └── no_nigiri/ (100+ fotos)

2. Ejecuta: python main.py
   ├── Carga todas las fotos
   ├── Entrena durante ~30 minutos
   ├── Guarda modelo_nigiri.h5
   └── Muestra la precisión

3. Ejecuta: streamlit run app.py
   └── Abre la interfaz en el navegador
```

### Después (Usar el modelo):

```
1. Ejecuta: streamlit run app.py
2. Sube una foto nueva
3. Recibe la predicción instantáneamente
```

---

## 📊 Métricas Clave Explicadas

### **Accuracy (Precisión)**
- **¿Qué es?** % de predicciones correctas
- **Ejemplo:** 90% significa que de 100 predicciones, 90 son correctas
- **¿Es bueno?** > 85% es muy bueno

### **Loss (Error)**
- **¿Qué es?** Cuánto se equivocó el modelo
- **Ejemplo:** Loss = 0.5 significa un error medio
- **¿Es bueno?** Cuanto más bajo, mejor (cercano a 0)

### **Confianza**
- **¿Qué es?** Qué tan segura está la predicción
- **Ejemplo:** 95% confianza = muy segura
- **¿Es importante?** Si es < 60%, puede no estar segura

---

## ⚠️ Problemas Comunes y Soluciones

### "El modelo no identifica bien"

**Causa:** Pocas imágenes o datos desequilibrados

**Solución:**
- Añade 100+ imágenes de nigiri de salmón
- Añade 100+ imágenes de otras cosas
- Asegúrate de tener números similares en ambas carpetas

### "Tarda mucho en entrenar"

**Causa:** Mucho volumen de imágenes

**Soluciones:**
- Usa imágenes más pequeñas (64x64 en lugar de 128x128)
- Reduce las épocas (20 en lugar de 50)
- Usa una GPU si tienes (es más rápido)

### "Dice que es nigiri cuando no lo es"

**Causa:** El modelo está confundido

**Soluciones:**
- Añade más imágenes de "no nigiri" variadas
- Las imágenes de "no nigiri" deben ser similares a nigiri (otros sushis, por ejemplo)

---

## 🛠️ Requisitos Técnicos Instalados

En `requirements.txt` tenemos:

- **TensorFlow/Keras:** La librería de IA que usa Google
- **NumPy:** Para manipular números y matrices
- **Pillow:** Para manejar imágenes
- **scikit-learn:** Para dividir datos
- **Streamlit:** Para crear la interfaz web

---

## 🎓 Analogías Finales para Entender Todo

### La máquina como estudiante:

| Concepto | Analogía |
|----------|----------|
| **Dataset** | Libros de texto |
| **Entrenar** | Estudiar |
| **Épocas** | Veces que estudia los mismos libros |
| **Predicción** | Responder un examen |
| **Accuracy** | Nota final |
| **Data Augmentation** | Practicar con ejercicios variados |
| **Dropout** | Estudiar sin apuntes a veces |
| **Early Stopping** | Dejar de estudiar cuando ya sabes |

---

## 📈 Mejoras Futuras

Para mejorar aún más el modelo:

1. **Más imágenes:** 500+ de cada categoría
2. **Transferencia de aprendizaje:** Usar un modelo pre-entrenado
3. **Mejor diversidad:** Diferentes iluminaciones, ángulos, fondos
4. **Validación cruzada:** Probar múltiples divisiones de datos

---

## 🎯 Resumen

Este proyecto automatiza algo que antes hacía solo un humano: distinguir entre nigiri de salmón y otras cosas. Lo hace:

1. **Mostrándole ejemplos** (entrenar con 100+ fotos)
2. **Aprendiendo patrones** (características que definen un nigiri)
3. **Practicando** (ajustando internamente hasta acertar)
4. **Generalizando** (reconociendo nuevas fotos nunca vistas)

¡Y todo en segundos! 🚀

---

**¿Preguntas?** Lee de nuevo la sección que no entiendas. La IA puede parecer magia, pero es solo matemáticas muy bien aplicadas. 🧮✨
