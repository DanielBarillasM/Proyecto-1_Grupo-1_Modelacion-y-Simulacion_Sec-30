# OUTBREAK: POISSON SURVIVAL

Simulación interactiva en **Streamlit + Python** de oleadas de enemigos en un videojuego de supervivencia. La llegada de enemigos se modela mediante un **Proceso de Poisson homogéneo** y los tiempos entre llegadas se generan con la **transformada inversa de la distribución Exponencial**.

## Idea matemática

Si `N(t)` es el número de enemigos que aparecen hasta el tiempo `t`, entonces:

\[
N(t) \sim \operatorname{Poisson}(\lambda t)
\]

Los tiempos entre llegadas son:

\[
\Delta_i \sim \operatorname{Exp}(\lambda)
\]

Y mediante el método de la inversa:

\[
\Delta_i = -\frac{\ln(U_i)}{\lambda}, \qquad U_i\sim U(0,1)
\]

Los instantes de aparición se construyen acumulando:

\[
t_i=t_{i-1}+\Delta_i
\]

## Características

- UI temática de apocalipsis.
- Arena táctica **3D interactiva** con Plotly.
- Parámetros de dificultad y combate ajustables.
- Visualización temporal de HP y enemigos activos.
- Gráfica del proceso de conteo `N(t)`.
- Comparación empírica de los interarribos con la densidad Exponencial teórica.
- Tabla de `U_i`, `Δ_i` y `t_i` para explicar el método del pizarrón.
- Simulación Monte Carlo de múltiples partidas.
- Curva `λ` vs. probabilidad de supervivencia.
- Semilla reproducible para repetir exactamente un escenario.
- Presentación HTML independiente, navegable con las flechas del teclado.

## Estructura

```text
zombie_poisson_streamlit/
├── app.py
├── simulation.py
├── visuals.py
├── requirements.txt
├── presentacion.html
├── README.md
└── .streamlit/
    └── config.toml
```

## Instalación

Desde una terminal, dentro de la carpeta del proyecto:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Linux / WSL / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit abrirá normalmente la aplicación en `http://localhost:8501`.

## Qué explicar durante la presentación

1. **Problema:** la aparición de enemigos es aleatoria, no periódica.
2. **Modelo:** `N(t)` es un Proceso de Poisson homogéneo de tasa `λ`.
3. **Variable continua:** los tiempos entre llegadas siguen una Exponencial.
4. **Generación:** se usa `Δ_i = -ln(U_i)/λ`.
5. **Acumulación:** `t_i = t_{i-1} + Δ_i`.
6. **Simulación del juego:** cada `t_i` crea un nuevo infectado; luego se simulan movimiento, combate y daño.
7. **Experimento:** se repite la partida con Monte Carlo para estimar la probabilidad de supervivencia.
8. **Conclusión:** al aumentar `λ`, aumenta la presión de la oleada y, en general, disminuye la probabilidad de supervivencia.

## Nota sobre la vista 3D

La vista 3D está construida con Plotly y representa el estado del modelo de simulación. No pretende sustituir un motor de videojuegos como Unity o Unreal; su objetivo es mantener todo el proyecto dentro de Streamlit/Python y conservar la relación directa entre el modelo matemático y la visualización.
