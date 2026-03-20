---
title: "B1 eRisk: Plan de Mejora hacia SOTA (F1 ≈ 0.60-0.71)"
description: "Roadmap para cerrar el gap entre F1=0.472 actual y SOTA=0.71"
---

## Estado Actual

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| F1 | 0.472 | ≥ 0.60 |
| AUC | 0.891 | ≥ 0.90 |
| Precision | 0.346 | ≥ 0.50 |
| Recall | 0.745 | ≥ 0.65 |

**Modelo**: SVC(C=0.1, RBF, gamma=0.01) sobre PCA-50 de MentalRoBERTa
**Split**: 2017 train (716) / 2017 val (171) / 2022 test (1398)
**Imbalance**: train 14.5% → test 7.0% (no data balancing, `class_weight='balanced'`)

**Diagnóstico**: AUC=0.891 indica que el ranking es bueno. El problema es precision (0.346) — demasiados FP. El modelo genera buenos scores pero el punto de corte no distingue bien en la distribución del 2022.

---

## Propuestas Ordenadas por Impacto Esperado

### P1: Anchor projections de B2 (DSM-5 dimensions)
**Impacto estimado**: Alto
**Esfuerzo**: Bajo (código ya existe en B2)

B2 proyecta los embeddings a 9 dimensiones clínicas (depressed mood, anhedonia, sleep, fatigue, etc.) + 1 healthy baseline = 10 dims. Estas dimensiones son **clínicamente definidas** y deberían ser más invariantes al temporal shift que los componentes PCA (que capturan varianza estadística, no semántica clínica).

**Acción**: Exportar las anchor features de B2 como parquet, cargarlas en B1_rigorous como feature set adicional. Probar `anchors`, `pca50+anchors`, `anchors+cvx`.

### P2: Calibración de probabilidades (Platt/isotonic)
**Impacto estimado**: Medio-Alto
**Esfuerzo**: Bajo

El threshold tuning en val ayudó (0.414→0.472), pero es discreto (granularidad 0.05). Calibración isotónica post-hoc adapta las probabilidades al prior del test set sin tocar los datos de test directamente. Esto es especialmente útil cuando la prevalencia cambia entre train y test.

**Acción**: `CalibratedClassifierCV(clf, method='isotonic', cv=5)` en train, evaluar en test.

### P3: Temporal feature engineering (CVX avanzado)
**Impacto estimado**: Medio
**Esfuerzo**: Medio

Las features temporales actuales (hurst, drift, velocity) no ayudan. Pero hay features CVX más ricas no exploradas:
- **Region trajectory**: distribución del usuario sobre clusters semánticos a lo largo del tiempo (`region_trajectory()`)
- **Wasserstein drift**: cambio distribucional entre primera y segunda mitad del historial (`wasserstein_drift()`)
- **Path signatures**: representación invariante a reparametrización de la trayectoria (`path_signature()`)
- **Changepoints**: número y severidad de cambios de régimen (`detect_changepoints()`)

**Acción**: Extraer estas features para cada usuario, añadir al feature set.

### P4: Incorporar eRisk 2018 al training
**Impacto estimado**: Alto
**Esfuerzo**: Medio

Con solo 716 users en train (104 dep), hay poco dato para generalizar. **Los datos de eRisk 2018 existen** en `data/eRisk/2018 (test cases)/` pero no están en los embeddings actuales.

Las ediciones de eRisk funcionan como matriusca: la edición N es el train del año N+1. Así que:
- **2017 train + 2017 test = train para 2018**
- **2018 test = nuevos subjects (solo test)**
- **2022 = edición separada con su propio test set**

Incorporar 2018 test subjects al training:
- Más usuarios → mejor generalización
- Reduce el gap temporal (2017→2022 se convierte en 2017+2018→2022)
- Split limpio: 2018 test es independiente de 2022 → no hay leakage

**Acción**:
1. Procesar los JSONs de eRisk 2018 test al formato unified
2. Generar MentalRoBERTa embeddings
3. Añadir al parquet con split='train'
4. Re-extraer features CVX con el pool ampliado
5. Re-ejecutar B1_erisk_rigorous con más datos

### P5: Ensemble de modelos
**Impacto estimado**: Medio
**Esfuerzo**: Medio

En vez de un solo modelo, combinar:
- SVC sobre PCA-50 (mejor generalización)
- XGBoost sobre raw 768d (mejor en val)
- LR sobre anchors (interpretable)

Stacking o voting con pesos calibrados en val.

### P6: Fine-tuning de MentalRoBERTa
**Impacto estimado**: Alto
**Esfuerzo**: Alto

Los embeddings actuales son fijos (frozen MentalRoBERTa). Fine-tuning el encoder con classification head sobre los datos de 2017 adaptaría los embeddings al task. Esto es lo que hacen los sistemas SOTA.

**Acción**: Fine-tune en HPC con la 3090. Requiere PyTorch training loop.

---

## Orden de Ejecución Recomendado

1. **P4 (eRisk 2018)** — más datos es lo que más impacta. Procesar 2018 test subjects y añadir al training pool
2. **P1 (Anchors)** — inmediato, código de B2 ya existe. DSM-5 dims son invariantes temporalmente
3. **P2 (Calibración)** — 10 líneas de código, ajusta probabilidades al prior real
4. **P3 (CVX features avanzadas)** — region_trajectory, wasserstein_drift, path_signatures
5. **P5 (Ensemble)** — después de tener P1-P4
6. **P6 (Fine-tuning)** — proyecto separado, mayor impacto

Con P4+P1+P2 deberíamos llegar al rango 0.55-0.65. Para 0.65+ probablemente se necesita P6 (fine-tuning).
