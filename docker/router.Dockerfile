FROM rust:1.98.1-slim-trixie AS build
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY proto ./proto
COPY router/Cargo.toml router/Cargo.lock ./router/
COPY router/build.rs ./router/build.rs
COPY router/src ./router/src
WORKDIR /build/router
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/router/target/release/dispatch-router /usr/local/bin/dispatch-router
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/dispatch-router"]
