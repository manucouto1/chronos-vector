# ── Build stage ───────────────────────────────────────────────────
FROM rust:1.88-bookworm AS builder

WORKDIR /app
COPY . .

RUN cargo build --release --bin cvx-server

# ── Runtime stage ────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgcc-s1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/cvx-server /usr/local/bin/cvx-server

ENV CVX_HOST=0.0.0.0
ENV CVX_PORT=3000
ENV RUST_LOG=info

EXPOSE 3000

ENTRYPOINT ["cvx-server"]
