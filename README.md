# Optimización de portafolio de Markowitz y análisis de riesgo

Aplica la **Teoría Moderna de Portafolios** a S&P 500, NASDAQ-100, oro y Bitcoin (2020–2025) y evalúa el portafolio óptimo con métricas de riesgo.

## Qué hace

- Matriz de correlación de retornos diarios (heatmap).
- Simulación de **10.000 portafolios** aleatorios y **frontera eficiente**.
- Portafolio de **máximo Sharpe**: pesos, rentabilidad y volatilidad anual.
- **Alpha y Beta** frente al S&P 500 con regresión OLS.
- **Value at Risk** histórico al 95% y 99%.
- Máximo profit y **drawdown** del último año.
- **Simulación Monte Carlo** de precios (movimiento browniano geométrico).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python MarcoWitz.py
```

## Limitaciones

- El Sharpe no resta la tasa libre de riesgo.
- Los pesos se optimizan con todo el histórico (dentro de la muestra).
- El Monte Carlo asume retornos normales y subestima eventos extremos.

## Autor

**Santiago Peña Díaz** — Estudiante de Ingeniería de Sistemas (CUN, Colombia)

- GitHub: [Lmao320](https://github.com/Lmao320)
- Documentación completa: [Portafolio en Notion](https://freezing-hope-5b5.notion.site/Portafolio-de-Proyectos-Santiago-Pe-a-D-az-41f7f18e4ca2811793dcece91c3d00c0)
