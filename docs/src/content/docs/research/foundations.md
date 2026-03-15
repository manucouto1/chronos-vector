---
title: "Theoretical Foundations"
description: "Literature review and research foundations for ChronosVector"
---

## 3. Literature Review: Fundamentos Teóricos

### 3.1 Embeddings Temporales y Diacrónicos

#### 3.1.1 Modelos Estáticos Slice-Based

El enfoque clásico entrena embeddings independientes por período temporal y luego los alinea. Yao et al. (2018) proponen el modelo D-EMBD (*Dynamic Word Embeddings*) que resuelve simultáneamente el aprendizaje time-aware y el problema de alineamiento, superando métodos que requieren "anchor words" o Orthogonal Procrustes como paso separado. Hamilton et al. (2016) establecen las "leyes" de conformidad e innovación que gobiernan las tasas de cambio semántico en función de la frecuencia y polisemia.

**Limitación:** Estos modelos discretizan el tiempo en "bins", perdiendo la resolución fina entre períodos.

#### 3.1.2 Modelos Continuos y Contextualizados

Trabajos recientes tratan los embeddings como **funciones continuas parametrizadas temporalmente**, ofreciendo alineamiento más suave y estimación de trayectorias de grano fino (Yao et al., 2017). Las aproximaciones basadas en transformers (HistBERT, embeddings contextualizados de Martinc et al., 2020) agregan representaciones contextuales para obtener representaciones específicas de cada período temporal.

La línea *sense-based* (Giulianelli et al., 2020; Montariol et al., 2021) trata los significados de cada palabra individualmente mediante clustering de embeddings contextualizados, distinguiéndose del enfoque *form-based* que analiza cómo cambia el significado dominante o el grado de polisemia.

#### 3.1.3 Temporal Knowledge Graph Embeddings (TKGE)

El campo de TKGE ha producido modelos sofisticados para capturar la evolución temporal de relaciones entre entidades:

- **TComplEx** (Lacroix et al., 2020): Extiende ComplEx a cuadrupletos (h, r, t, τ) mediante descomposición tensorial de 4º orden con regularización temporal.
- **TPComplEx** (Yang et al., 2024): Mejora TComplEx modelando tres propiedades del "time perspective": simultaneidad, agregación y asociatividad.
- **TempHypE** (Bhullar & Kobti, 2026): Combina geometría hiperbólica (Poincaré ball) con Neural ODEs para link prediction temporal, logrando +43% MRR sobre baselines.
- **TeRo** (Xu et al., 2020): Codifica timestamps como rotaciones que afectan entidades en espacio complejo.
- **ChronoR** (Sadeghian et al., 2021): Representa la interacción relación-timestamp como rotaciones conectando embeddings de entidades.

**Insight para CVX:** La representación como cuadrupleto $(entity, relation, entity, time)$ es directamente mapeable a nuestra estructura `TemporalPoint<V, T>`.

### 3.2 Neural Ordinary Differential Equations

Chen et al. (2018) introducen Neural ODEs, donde en lugar de capas discretas, el estado oculto evoluciona según $\frac{dh}{dt} = f_\theta(h(t), t)$ con $f_\theta$ parametrizada por una red neuronal. La salida se computa mediante un solver ODE black-box, con backpropagation a través del *adjoint method* (sin almacenar activaciones intermedias, coste de memoria constante).

**Variantes relevantes para CVX:**

- **Latent ODEs** (Rubanova et al., NeurIPS 2019): Modelan series temporales irregularmente muestreadas en un espacio latente continuo. El encoder (ODE-RNN) procesa observaciones hacia atrás; el decoder integra la ODE latente hacia adelante. Ideal para nuestro caso de vectores que llegan a intervalos irregulares.
- **Neural CDEs** (Kidger et al., 2020): Controlled Differential Equations — generalizan RNNs al caso continuo con entradas como paths controlados.
- **Graph Neural ODEs** (Poli et al., AAAI 2020): Extienden GCN a tiempo continuo, donde los embeddings de nodos evolucionan como un sistema dinámico continuo. Aplicable si las relaciones entre entidades también evolucionan.

Zhang et al. (2025) publican un survey comprehensivo de Neural ODEs que cataloga los desafíos abiertos: capacidad de aproximación limitada, sensibilidad a inputs adversariales, y alto coste computacional. Proponen direcciones futuras incluyendo mejoras en estabilidad numérica y eficiencia del solver.

**Insight para CVX:** La predicción $v(t + \Delta t) \approx v(t) + \int f_\theta(v, t) dt$ es el corazón de nuestro motor de extrapolación. En Rust, podemos implementar solvers ODE adaptativos (Dormand-Prince/RK45) con SIMD para las evaluaciones de $f_\theta$.

### 3.3 Geometrías No-Euclidianas para Embeddings

#### 3.3.1 Embeddings Hiperbólicos

Nickel & Kiela (NeurIPS 2017) demuestran que el espacio hiperbólico — específicamente el modelo del Poincaré ball $\mathbb{B}^d$ — captura simultáneamente jerarquía y similaridad con representaciones mucho más parsimoniosas que las euclidianas. La distancia hiperbólica:

$$d_{\mathbb{B}}(u, v) = \text{arcosh}\left(1 + 2\frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

crece exponencialmente cerca del borde del ball, permitiendo que árboles con factor de ramificación $b$ se embeden con distorsión mínima.

**¿Por qué importa para CVX?** Las evoluciones temporales de conceptos forman naturalmente jerarquías: un concepto se especializa (*narrowing*), se generaliza (*broadening*), o se bifurca en sub-sentidos. El espacio hiperbólico puede capturar estas trayectorias con muchas menos dimensiones que el espacio euclídeo.

#### 3.3.2 Embeddings Hiperbólico-Temporales

TempHypE (Bhullar & Kobti, ASONAM 2025) une geometría hiperbólica con Neural ODEs para knowledge graphs temporales, logrando +43% MRR en ICEWS18 y +30% Hits@10 sobre baselines euclidianas. Esta línea de investigación sugiere que CVX debería soportar **métricas configurables**: euclídea, coseno, hiperbólica y Lorentziana, según la naturaleza de los datos.

### 3.4 Búsqueda Vectorial a Escala

#### 3.4.1 HNSW (Hierarchical Navigable Small World)

Malkov & Yashunin (2018) presentan HNSW como una estructura de grafos multi-capa donde la capa superior es dispersa (links largos para navegación rápida) y la inferior contiene todos los vectores (links cortos para precisión). La búsqueda combina descenso jerárquico con beam search en la capa base, logrando complejidad $O(\log N)$.

**Limitación temporal:** HNSW asume un conjunto estático de vectores. Wang et al. (ICDE 2025) demuestran que construir un HNSW separado por timestamp tiene complejidad $O(MN\log N)$ en tiempo de actualización — prohibitivo para aplicaciones dinámicas.

#### 3.4.2 DiskANN y Vamana

Subramanya et al. (NeurIPS 2019) presentan DiskANN, que combina el algoritmo Vamana para construir grafos SSD-friendly con Product Quantization en RAM para búsquedas rápidas. Logra 95%+ recall con latencia <5ms sobre mil millones de vectores en una sola máquina, usando 32 GB RAM + SSD frente a los 500 GB que requeriría HNSW.

**Evolución reciente:** DiskANN ha sido **reescrito en Rust** por Microsoft (2023+), con una arquitectura de "orquestador stateless" que delega almacenamiento a un backend vía Provider API. FreshDiskANN añade soporte para inserciones/borrados en tiempo real manteniendo >95% recall.

#### 3.4.3 TANNS: Búsqueda Temporal Nativa

Wang et al. (ICDE 2025) proponen el **Timestamp Graph**, una estructura que explota la localidad temporal:

- **Estructura:** Un grafo de proximidad unificado que gestiona vectores válidos a través de todos los timestamps. Cada nodo tiene listas de vecinos versionadas temporalmente.
- **Backup Neighbors:** Para manejar la expiración de puntos sin degradar la conectividad del grafo.
- **Historic Neighbor Tree (HNT):** Comprime las listas de vecinos históricas como árboles binarios, reduciendo el almacenamiento de $O(M^2N)$ a $O(MN)$ con las mismas garantías de búsqueda.

**Insight para CVX:** TANNS es el estado del arte más directamente aplicable a nuestro índice ST-HNSW. Debemos integrar estas ideas como capa base y añadir nuestras innovaciones (decay, predicción) encima.

### 3.5 Detección de Change Points

#### 3.5.1 Métodos Offline

**PELT** (Pruned Exact Linear Time, Killick et al., 2012): Algoritmo de detección exacta de cambios múltiples con complejidad $O(N)$ bajo la condición de que el número de change points crece linealmente con los datos. Usa programación dinámica con pruning: un timestamp $t$ se descarta como candidato si el coste mínimo de segmentación hasta $t$ más el coste del segmento $[t, s]$ excede el coste óptimo hasta $s$. El criterio de penalización (BIC, AIC) controla la sensibilidad.

**Wild Binary Segmentation (WBS)** (Fryzlewicz, 2014): Detecta change points muestreando sub-intervalos aleatorios y agregando resultados, útil cuando el ruido dificulta la localización precisa.

**GeomFPOP** (Pishchagina et al., 2024): Extiende el *functional pruning* a series temporales multivariadas usando formas geométricas (esferas e hiperrectángulos) para la poda, logrando complejidad casi-lineal independiente del número de cambios.

#### 3.5.2 Métodos Online

**BOCPD** (Bayesian Online Change Point Detection, Adams & MacKay, 2007): Computa recursivamente la distribución posterior del "run length" (tiempo desde el último change point) a cada nueva observación. Particularmente adecuado para datos en streaming con interpretación probabilística. Fan & Mackey (2017) extienden BOCPD al caso multi-secuencia para analizar cambios en 401 acciones del S&P 500 simultáneamente.

BOCPD ha sido recientemente extendido para clasificar cambios (anomalías colectivas vs. change points genuinos, Arxiv 2025), con mejoras de +35% en F1 para detección de anomalías colectivas online.

#### 3.5.3 Drift en Embeddings

La detección de drift en embeddings típicamente compara la similaridad coseno promedio entre representaciones de dos ventanas temporales. Cuando la similaridad cae por debajo de un umbral, se señala drift semántico. Este enfoque es más simple que los métodos estadísticos pero captura cambios sutiles en cómo los conceptos se relacionan — por ejemplo, cómo "stream" pasó de significar "flujo de vídeo" a incluir "streaming de música".

**Insight para CVX:** Implementar PELT para análisis offline y BOCPD para monitoreo online, con la capacidad de operar directamente sobre las trayectorias vectoriales almacenadas.

### 3.6 Compresión Vectorial

#### 3.6.1 Product Quantization (PQ)

Jégou et al. (2011) proponen PQ: dividir cada vector de dimensión $D$ en $M$ sub-vectores de dimensión $D/M$, entrenar $K$ centroides por sub-espacio vía k-means, y representar cada vector como $M$ índices de centroide. Con $M=8$ y $K=256$, un vector de 128 dimensiones (512 bytes en FP32) se comprime a 8 bytes — **97% de reducción**. La búsqueda usa Asymmetric Distance Computation (ADC): distancias exactas del query a centroides, luego lookup table.

#### 3.6.2 Additive Quantization (AQ)

Babenko & Lempitsky (CVPR 2014) generalizan PQ: en lugar de descomponer en sub-espacios ortogonales, cada codebook contiene codewords de dimensión completa $D$. Un vector se aproxima como la **suma** de $M$ codewords, eliminando la asunción de independencia entre sub-espacios. AQ logra menor error de aproximación que PQ con la misma longitud de código.

#### 3.6.3 Quantización Temporal

**Innovación propuesta para CVX:** Delta-Quantization temporal — almacenar solo $\Delta v = v(t_i) - v(t_{i-1})$ cuando el delta supera un umbral de relevancia. Los deltas tienden a ser sparse (pocas dimensiones cambian significativamente entre updates consecutivos), permitiendo una compresión adicional mediante codificación sparse + PQ sobre los deltas.

---

## 7. Algorithms & Research References

### 7.1 Referencias Fundacionales

1. **Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D.** (2018). *Neural Ordinary Differential Equations*. NeurIPS 2018 (Best Paper Award). [arXiv:1806.07366]
   — Fundamento teórico para modelar evolución de embeddings como flujo continuo.

2. **Rubanova, Y., Chen, R.T.Q., & Duvenaud, D.** (2019). *Latent Ordinary Differential Equations for Irregularly-Sampled Time Series*. NeurIPS 2019.
   — Clave para manejar vectores que llegan a intervalos irregulares.

3. **Nickel, M. & Kiela, D.** (2017). *Poincaré Embeddings for Learning Hierarchical Representations*. NeurIPS 2017.
   — Base para métricas hiperbólicas en CVX.

4. **Malkov, Y. & Yashunin, D.** (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. IEEE TPAMI.
   — Algoritmo base para nuestro índice ST-HNSW.

### 7.2 Búsqueda Vectorial a Escala

5. **Subramanya, S.J., Devvrit, F., Simhadri, H.V., Krishnaswamy, R., & Kadekodi, R.** (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*. NeurIPS 2019.
   — Algoritmo Vamana y estrategia SSD + PQ en RAM.

6. **Singh, A., Subramanya, S.J., Krishnaswamy, R., & Simhadri, H.V.** (2021). *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search*. arXiv:2105.09613.
   — Actualizaciones en tiempo real sin reconstruir el índice.

7. **Wang, H., Wu, H., et al.** (2025). *Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data*. IEEE ICDE 2025.
   — Timestamp Graph y Historic Neighbor Tree — referencia directa para nuestra capa de indexación temporal.

8. **DiskANN Rust Rewrite.** Microsoft (2023–presente). [github.com/microsoft/DiskANN]
   — Reescritura completa en Rust con arquitectura stateless y Provider API.

### 7.3 Embeddings Temporales y Semántica Diacrónica

9. **Yao, Z., Sun, Y., Ding, W., Rao, N., & Xiong, H.** (2018). *Dynamic Word Embeddings for Evolving Semantic Discovery*. ACM WSDM 2018.
   — Modelo D-EMBD con alineamiento temporal simultáneo.

10. **Hamilton, W.L., Leskovec, J., & Jurafsky, D.** (2016). *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change*. ACL 2016.
    — Leyes de conformidad e innovación; metodología de evaluación temporal.

11. **Periti, F. & Montanelli, S.** (2024). *Studying Word Meaning Evolution through Incremental Semantic Shift Detection*. Language Resources and Evaluation, Springer.
    — WiDiD: clustering evolutivo para detección incremental de cambio semántico.

12. **Kutuzov, A., Øvrelid, L., Szymanski, T., & Velldal, E.** (2018). *Diachronic Word Embeddings and Semantic Shifts: A Survey*. COLING 2018.
    — Taxonomía comprehensiva de tipos de cambio semántico y métodos de medición.

13. **Couto, M., Perez, A., Parapar, J., & Losada, D.** (2025). *Temporal Word Embeddings for Early Detection of Psychological Disorders on Social Media*. Journal of Healthcare Informatics Research.
    — Aplicación directa de embeddings temporales a dominios sensibles.

### 7.4 Temporal Knowledge Graph Embeddings

14. **Lacroix, T., Obozinski, G., & Usunier, N.** (2020). *Tensor Decompositions for Temporal Knowledge Graph Completion*. EMNLP 2020.
    — TComplEx y TNTComplEx.

15. **Bhullar, A. & Kobti, Z.** (2026). *TempHypE: Time-Aware Hyperbolic Neural ODE Knowledge Graph Embeddings for Dynamic Link Prediction*. ASONAM 2025 / LNCS 16324.
    — Convergencia de geometría hiperbólica + Neural ODEs para temporal KG.

16. **Yang, J., et al.** (2024). *Tensor Decompositions for Temporal Knowledge Graph Completion with Time Perspective*. Expert Systems with Applications.
    — TPComplEx: perspectiva temporal con simultaneidad, agregación, asociatividad.

17. **Goel, R., Kazemi, S.M., Brubaker, M., & Poupart, P.** (2020). *Diachronic Embedding for Temporal Knowledge Graph Completion*. AAAI 2020.
    — DE-SimplE: embeddings diacrónicos para TKG.

### 7.5 Detección de Drift y Change Points

18. **Killick, R., Fearnhead, P., & Eckley, I.A.** (2012). *Optimal Detection of Changepoints with a Linear Computational Cost*. Journal of the American Statistical Association.
    — Algoritmo PELT.

19. **Adams, R.P. & MacKay, D.J.C.** (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
    — BOCPD fundacional.

20. **Hinder, F., Vaquet, V., & Hammer, B.** (2024). *One or Two Things We Know about Concept Drift — A Survey on Monitoring in Evolving Environments. Part A: Detecting Concept Drift*. Frontiers in AI, 7, 1330257.
    — Taxonomía más completa de detección de drift no supervisada.

21. **Hinder, F., Vaquet, V., & Hammer, B.** (2024). *Part B: Locating and Explaining Concept Drift*. Frontiers in AI, 7, 1330258.
    — Localización y explicación de drift — permite entender *qué* cambió, no solo *cuándo*.

22. **Pishchagina, O., et al.** (2024). *Geometric-Based Pruning Rules for Change Point Detection in Multiple Independent Time Series*. Computo Journal.
    — GeomFPOP: extensión de PELT a series multivariadas.

23. **Lukats, D., Zielinski, O., Hahn, A., & Stahl, F.** (2024). *A Benchmark and Survey of Fully Unsupervised Concept Drift Detectors on Real-World Data Streams*. International Journal of Data Science and Analytics, 19, 1-31.
    — Benchmark de 10 algoritmos unsupervised sobre 11 datasets reales.

### 7.6 Compresión y Quantización

24. **Jégou, H., Douze, M., & Schmid, C.** (2011). *Product Quantization for Nearest Neighbor Search*. IEEE TPAMI.
    — PQ fundacional.

25. **Babenko, A. & Lempitsky, V.** (2014). *Additive Quantization for Extreme Vector Compression*. CVPR 2014.
    — AQ: generalización de PQ sin asunción de sub-espacios ortogonales.

### 7.7 Neural ODEs — Surveys y Extensiones

26. **Zhang, B., Murshed, M., Cheng, Z., & Luo, W.** (2025). *A Survey on Neural Ordinary Differential Equations*. Springer LNCS.
    — Survey comprehensivo de desafíos y direcciones futuras de Neural ODEs.

27. **Biloš, M., et al.** (2021). *Neural Flows: Efficient Alternative to Neural ODEs*. NeurIPS 2021.
    — Alternativa más eficiente a Neural ODEs para ciertos escenarios.

28. **Poli, M., Massaroli, S., Park, J., Yamashita, A., Asama, H., & Park, J.** (2020). *Graph Neural Ordinary Differential Equations*. AAAI DLGMA 2020.
    — Extensión de ODEs a datos de grafo, relevante para relaciones entre entidades.

### 7.8 Vector Databases en Rust

29. **Qdrant.** Vector Database built in Rust with SIMD and custom Gridstore engine. [qdrant.tech]
    — Referencia de arquitectura: quantización hasta 64x, HNSW con updates en tiempo real.

30. **Lance / LanceDB.** Columnar vector storage en Rust basado en formato Lance (evolución de Parquet para ML). [lancedb.com]
    — Referencia para warm storage columnar optimizado para vectores.
