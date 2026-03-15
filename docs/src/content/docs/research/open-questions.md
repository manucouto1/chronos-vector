---
title: "Open Research Questions"
description: "Unsolved problems and research directions for ChronosVector"
---

## 11. Open Research Questions

1. **¿Cuál es la métrica óptima para combinar distancia semántica y temporal?** La combinación lineal $\alpha \cdot d_{sem} + (1-\alpha) \cdot d_{time}$ es simple pero puede no capturar interacciones no-lineales.

2. **¿Cómo escala BOCPD a miles de streams vectoriales simultáneos?** La complejidad por-stream es $O(1)$ amortizada, pero el overhead total puede ser significativo.

3. **¿Los embeddings hiperbólicos mejoran la representación de trayectorias temporales en la práctica?** TempHypE muestra resultados prometedores en KGs pero no se ha probado en vectores de embedding general.

4. **¿Cuál es el ratio óptimo de keyframes a deltas?** Similar al trade-off GOP en video coding, pero en espacio vectorial de alta dimensión.

5. **¿Puede un Neural ODE ligero (pocas capas, inferencia rápida) predecir trayectorias útilmente?** El balance entre complejidad del modelo y latencia de query es crítico.

## Stochastic Analytics

6. **¿Qué profundidad de path signature es óptima?** Depth 2 captura drift y volatilidad. Depth 3+ captura patrones más complejos pero crece exponencialmente. ¿Cuál es el sweet spot para embedding trajectories?

7. **¿Neural SDE vs Neural CDE?** Los Neural CDEs (Kidger et al., 2020) manejan mejor datos irregulares que los SDEs. ¿Merece la complejidad adicional?

8. **¿Los embeddings exhiben mean reversion?** Si los embeddings de conceptos estables mean-revert, el modelo Ornstein-Uhlenbeck es apropiado. Si no, un random walk o trending model es mejor. Requiere validación empírica.

## Implementation & Integration

9. **¿Cómo escalan las path signatures a D=768?** La computación directa sobre 768 dimensiones es intratable. La reducción via PCA a 5-10 dims pierde información. ¿Cuánta?

10. **¿El flow imbalance (coherencia de drift del vecindario) es predictivo?** Concepto inspirado en microestructura financiera. Necesita validación empírica en embeddings.

11. **¿Modelo version alignment funciona en la práctica?** Procrustes alignment entre model v1 y v2 asume una transformación lineal. ¿Es suficiente cuando los modelos cambian significativamente?

---

## 8. Development Roadmap (The "Rustacean" Path)

### Phase 1: The Core (Meses 1–3)

- Implementar `TemporalPoint<V, T>` genérico sobre `V: VectorSpace` y `T: Timestamp`.
- Trait `DistanceMetric` con implementaciones SIMD para coseno, L2, dot product.
- Índice HNSW básico con filtrado por rango de tiempo usando Roaring Bitmaps.
- Storage en RocksDB con key prefix temporal.
- **Deliverable:** Benchmark comparativo vs Qdrant en queries snapshot kNN.

### Phase 2: Temporal Intelligence (Meses 4–6)

- Integrar **Timestamp Graph** (TANNS) para búsquedas por timestamp exacto.
- Implementar **delta-embedding storage** con keyframes periódicos.
- Calcular **vector velocity** ($\partial v / \partial t$) como primera derivada numérica.
- Implementar **PELT** en Rust para change point detection offline.
- **Deliverable:** Demo de evolutionary path + velocity queries.

### Phase 3: The API & Persistence (Meses 7–9)

- API HTTP con `axum` (REST) y `tonic` (gRPC streaming).
- Formato de archivo `.tvx` (Temporal Vector Index) para persistencia.
- **Tiered storage**: hot (RocksDB) → warm (Parquet/arrow-rs) → cold (Object Store + PQ).
- Integración con `object_store` crate para S3/MinIO.
- **Deliverable:** API funcional con ingest streaming y queries temporales.

### Phase 4: Neural Engine (Meses 10–12)

- Implementar solver ODE adaptativo (Dormand-Prince RK45) con SIMD.
- Integrar `burn` para Neural ODE inference: $f_\theta$ como MLP pequeño.
- **BOCPD online** para detección de drift en tiempo real durante ingesta.
- Queries de **extrapolation** y **temporal analogy**.
- **Deliverable:** Motor de predicción funcional con benchmarks de accuracy.

### Phase 5: Scale & Distribute (Meses 13–15)

- **DiskANN-style partitioning** temporal para datasets >RAM.
- Sharding temporal con `openraft` para consistencia.
- Métricas hiperbólicas (Poincaré ball) como alternativa configurable.
- **Dashboard de drift monitoring** (web UI simple).
- **Deliverable:** Benchmark a escala billón con métricas de latencia y recall.

### Phase 6: Research Frontier (Ongoing)

- Embeddings hiperbólico-temporales (estilo TempHypE).
- Latent ODE encoder para series temporales irregulares.
- Quantización adaptativa de deltas.
- Publicación de paper técnico describiendo la arquitectura.
