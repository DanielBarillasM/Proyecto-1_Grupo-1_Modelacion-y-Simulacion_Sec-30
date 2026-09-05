# Registro de auditoría y tareas

Este archivo conserva la trazabilidad de la auditoría contra
`Proyecto_1_Modelación_y_simulación.pdf`. Los elementos marcados como
completados forman parte de la entrega actual; las mejoras opcionales no son
requisitos pendientes de la rúbrica.

## Cumplimiento de la rúbrica

| Criterio del PDF (5 puntos cada uno) | Estado verificable | Evidencia principal |
|---|---|---|
| Descripción del sistema | Completo | README e informe, sección de sistema real |
| Modelo | Completo | Poisson, exponencial, renovación lognormal y fuentes aleatorias |
| Método de simulación | Completo | Inversa, Marsaglia Polar, Box--Muller y motor de paso fijo |
| Objetivos | Completo | Objetivo general y siete objetivos específicos en el informe |
| Resultados | Completo | CSV, macros y resultados de 400 pares reproducibles |
| Análisis | Completo | Wilson, Holm, McNemar, Wilcoxon, bootstrap y matriz de decisión |
| Resultados visuales | Completo | Figuras regenerables, tablas y curvas de sensibilidad |
| Multimedia o demostración | Completo | Aplicación Streamlit y escena Plotly 3D interactiva |
| Exposición oral | Preparada; depende del equipo | Guion de 17 minutos dividido entre cuatro integrantes |
| Presentación visual | Completo | 26 diapositivas HTML offline con MathJax e imágenes locales |

Los nueve criterios que pueden auditarse desde el repositorio tienen evidencia
completa. El puntaje de exposición oral no puede garantizarse mediante código:
dependerá de que el equipo ensaye, respete tiempos y responda preguntas. El
repositorio sí aporta el guion, reparto y ruta de demostración necesarios.

## Correcciones aplicadas en la auditoría de septiembre de 2026

- [x] Recalibrar el escenario base a 55.8 llegadas/min, CV 0.60 y 9.5 HP/s
  para evitar resultados saturados cerca de 100 %.
- [x] Regenerar los 400 pares del escenario base, curvas, macros y CSV.
- [x] Añadir comparación reproducible Marsaglia Polar--Box--Muller con KS,
  momentos, benchmark, rechazo y matriz multicriterio.
- [x] Corregir la interpretación de multiplicidad: el 60.3 % es ilustrativo
  bajo independencia; Holm no necesita ese supuesto.
- [x] Eliminar la recomendación incorrecta de buscar una semilla favorable.
- [x] Hacer que `Poisson check/comparacion_poisson_polar.py` escriba en la
  carpeta versionada correcta.
- [x] Sincronizar README, informe, presentación HTML y guion con los resultados.
- [x] Versionar las fuentes de la ficha DOCX/PDF y su dependencia opcional.
- [x] Ampliar la suite a 28 pruebas e incluir la matriz de decisión.

## Reglas para mantener la entrega sincronizada

Ejecutar desde `zombie_poisson_streamlit` cuando cambie el motor o un parámetro:

```powershell
python .\report\generate_report_assets.py
python .\presentation\generate_ficha.py
python -m unittest discover -s .\tests -v
```

Después, compilar dos veces `report/informe.tex`,
`presentation/ficha_repositorio.tex` y
`presentation/guion_exposicion.tex` con `pdflatex`.

## Mejoras futuras opcionales

- [ ] Calibrar llegadas y atributos con telemetría observada.
- [ ] Evaluar un proceso de Poisson no homogéneo con tasa `lambda(t)`.
- [ ] Incorporar movimiento del protagonista y colisiones funcionales.
- [ ] Construir una versión de escritorio 3D en Panda3D si cambia el alcance.

Estas extensiones no bloquean el cumplimiento actual. La conclusión sobre qué
modelo representa mejor un videojuego real sí quedaría pendiente de datos:
sin telemetría, Poisson es la hipótesis principal por parsimonia y no una
verdad empírica demostrada.
