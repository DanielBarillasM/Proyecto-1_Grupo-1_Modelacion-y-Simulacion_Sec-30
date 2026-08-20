# Outbreak: Stochastic Survival Lab

Proyecto académico de Modelación y Simulación desarrollado con Python y
Streamlit. El sistema representa una misión de supervivencia ante infectados,
modela sus instantes de aparición y estima la probabilidad de que el protagonista
alcance el horizonte temporal con vida.

La entrega compara dos alternativas con el mismo tiempo medio entre llegadas:

- un proceso de Poisson homogéneo con interarribos exponenciales;
- un proceso de renovación con interarribos lognormales generados mediante el
  método polar de Marsaglia.

Esta distinción es importante: Poisson es un modelo de conteo y Polar es un
método de generación de variables normales. La comparación no afirma que ambos
sean el mismo tipo de objeto matemático.

## Objetivo

Modelar la llegada aleatoria de infectados, integrar esas llegadas con reglas de
movimiento y combate, y estudiar cómo la tasa y la variabilidad temporal afectan:

- la probabilidad de supervivencia;
- el tiempo resistido;
- el número de infectados generados y eliminados;
- la cantidad máxima de enemigos activos simultáneamente.

La pregunta central es: **¿cómo cambia la supervivencia cuando aumenta la tasa de
llegadas o cambia la distribución de los interarribos?**

## Modelos matemáticos

### Proceso de Poisson

Sea \(N(t)\) el número de llegadas hasta el instante \(t\). Para una tasa
constante \(\lambda\):

\[
N(t)\sim\operatorname{Poisson}(\lambda t),
\qquad
P\{N(t)=k\}=e^{-\lambda t}\frac{(\lambda t)^k}{k!}.
\]

Por tanto:

\[
\mathbb E[N(t)]=\operatorname{Var}[N(t)]=\lambda t.
\]

Los interarribos son variables continuas exponenciales. Se generan con
transformada inversa:

\[
U_i\sim U(0,1),
\qquad
\Delta_i=-\frac{\ln(U_i)}{\lambda}.
\]

### Alternativa Polar-lognormal

El método polar de Marsaglia transforma dos uniformes en normales estándar. Se
toman \(V_1,V_2\sim U(-1,1)\), se calcula \(S=V_1^2+V_2^2\) y se acepta el par
cuando \(0<S<1\):

\[
Z=V_1\sqrt{\frac{-2\ln(S)}{S}}\sim N(0,1).
\]

Una normal no puede usarse directamente como tiempo porque admite valores
negativos. Por eso se construye un interarribo lognormal:

\[
\Delta_i=e^{\mu+\sigma Z_i}.
\]

Si \(c\) es el coeficiente de variación deseado:

\[
\sigma^2=\ln(1+c^2),
\qquad
\mu=\ln(1/\lambda)-\frac{\sigma^2}{2}.
\]

Con esta parametrización, \(\mathbb E[\Delta]=1/\lambda\). Los dos modelos
comparten frecuencia media, pero no la forma ni la propiedad de falta de memoria.

### Instantes de llegada

En ambos modelos:

\[
t_0=0,
\qquad
t_i=t_{i-1}+\Delta_i.
\]

Cada \(t_i\) crea un infectado. El combate se aproxima con un paso temporal fijo
\(dt\).

## Reglas de la simulación

1. El protagonista permanece en el centro de una arena circular.
2. Los infectados aparecen cerca del perímetro y avanzan radialmente.
3. El protagonista ataca al enemigo vivo más cercano dentro del alcance.
4. Cada infectado en contacto aporta daño de forma concurrente.
5. Hay victoria si se alcanza el tiempo objetivo con HP positivo.
6. Hay derrota si el HP llega a cero.

Los atributos enemigos incorporan variación uniforme acotada alrededor de sus
valores base. Las fuentes aleatorias de llegadas y atributos están separadas, de
modo que una comparación con la misma semilla conserve atributos equivalentes
para el enemigo de igual índice.

## Parámetros

| Parámetro | Unidad | Valor inicial | Interpretación |
|---|---:|---:|---|
| Duración | s | 90 | Horizonte de la misión |
| Tasa | llegadas/min | 39 | Se convierte a \(\lambda=0.65\) por segundo |
| CV Polar | adimensional | 0.45 | Variabilidad de interarribos lognormales |
| HP protagonista | HP | 110 | Vida disponible |
| DPS protagonista | HP/s | 42 | Capacidad ofensiva |
| Alcance | m | 10 | Radio efectivo del arma |
| HP infectado | HP | 42 | Resistencia base |
| Velocidad infectado | m/s | 1.55 | Avance radial base |
| DPS infectado | HP/s | 9 | Daño base al contactar |
| Paso temporal | s | 0.05 | Resolución numérica |

La interfaz expone las unidades y evita combinaciones geométricas inválidas. El
motor también valida valores positivos, relaciones entre radios y límites de
precisión.

## Inferencia Monte Carlo

Para \(R\) partidas independientes:

\[
\widehat p=\frac{1}{R}\sum_{r=1}^{R}I_r,
\]

donde \(I_r=1\) si la partida \(r\) termina con supervivencia. La aplicación no
muestra solamente el estimador puntual: calcula un intervalo de Wilson al 95 %.

La comparación Poisson--Polar utiliza las mismas semillas de corrida. Esto es un
diseño pareado para disminuir variación ajena al modelo de llegadas.

## Interfaz y visualización 3D

La escena tridimensional usa `plotly.graph_objects.Mesh3d`. Los personajes se
construyen proceduralmente con mallas de bajo poligonaje y el escenario contiene
terreno, edificios en ruinas, vehículos, barricadas y anillos tácticos. Todo se
mantiene en el navegador y se puede rotar, acercar y explorar por tiempo.

No se integró Pygame porque su abstracción principal es una superficie y un bucle
de eventos de escritorio; para 3D necesita una capa adicional como OpenGL y no se
embebe naturalmente en el ciclo reactivo de Streamlit. [La documentación oficial
de Pygame](https://www.pygame.org/docs/) presenta superficies, sprites y bucle de
juego, y también indica que puede funcionar como capa de visualización para
PyOpenGL.

Para una segunda aplicación independiente, [Panda3D](https://docs.panda3d.org/1.10/python/introduction/index)
sí es un motor 3D completo con enlaces de Python, escena, animación y bucle de
renderizado. En esta entrega se priorizó la integración web y analítica. Plotly
documenta la construcción explícita de triángulos mediante `i`, `j` y `k` en
[3D Mesh Plots](https://plotly.com/python/3d-mesh/).

## Estructura

```text
zombie_poisson_streamlit/
|-- .streamlit/
|   `-- config.toml
|-- assets/
|   |-- outbreak-command-center.png
|   `-- outbreak-tactical-model.png
|-- tests/
|   `-- test_simulation.py
|-- app.py
|-- simulation.py
|-- visuals.py
|-- informe.tex
|-- informe.pdf
|-- presentacion.html
|-- requirements.txt
`-- README.md
```

- `simulation.py`: generadores, combate, Monte Carlo e intervalos de confianza.
- `visuals.py`: escena 3D procedural y gráficas estadísticas.
- `app.py`: estado, formularios, explicación matemática y experimentos.
- `tests/test_simulation.py`: pruebas deterministas y estadísticas.
- `informe.tex`: informe académico en LaTeX.
- `presentacion.html`: presentación navegable con teclado y controles.

Las imágenes `assets/outbreak-command-center.png` y
`assets/outbreak-tactical-model.png` fueron generadas para este proyecto con una
herramienta de generación de imágenes de OpenAI. No contienen marcas ni material
de una franquicia reconocible.

## Instalación

Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

En Linux, WSL o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre normalmente en `http://localhost:8501`.

## Uso recomendado

1. Seleccionar el modelo y configurar la tasa en llegadas por minuto.
2. Revisar atributos de combate y semilla.
3. Pulsar **SIMULAR MISIÓN**.
4. Explorar la escena con el control temporal.
5. Revisar el desarrollo matemático y las variables generadas.
6. Ejecutar la comparación pareada.
7. Ejecutar Monte Carlo y discutir el intervalo de confianza.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Las pruebas cubren reproducibilidad, media y varianza del conteo Poisson, media
de los interarribos lognormales, coherencia de eventos después de la derrota,
validación de geometría, emparejamiento de semillas y estabilidad respecto a
`dt`.

## Informe

Con una distribución de LaTeX instalada:

```powershell
pdflatex informe.tex
pdflatex informe.tex
```

La segunda ejecución actualiza referencias y tabla de contenido.

## Supuestos y limitaciones

- La tasa es constante dentro de cada partida.
- Los infectados no colisionan ni evitan obstáculos; las ruinas son visuales.
- El protagonista no se desplaza y enfoca un objetivo a la vez.
- El combate es una aproximación de tiempo discreto.
- La curva de supervivencia es una estimación Monte Carlo, no una solución
  analítica exacta.
- Un histograma de una partida puede estar sesgado por el horizonte de
  observación; se utiliza como diagnóstico visual y no como prueba definitiva.
- Poisson es apropiado si tasa constante, independencia y falta de memoria son
  razonables. No es universalmente superior a cualquier proceso alternativo.

## Conclusión esperada

Poisson es la opción principal por su interpretación, parsimonia y relación
directa entre conteo Poisson e interarribos exponenciales. El modelo
Polar-lognormal sirve como contraste: demuestra que conservar la misma tasa media
no conserva necesariamente las rachas, la concurrencia ni la probabilidad de
supervivencia. La afirmación final debe apoyarse en los resultados y sus intervalos
de confianza obtenidos durante la ejecución.
