# Inference image for the reinterpretation service: HepMC in, SR efficiency out.
#   docker build -t jet-surrogate .
#   docker run --rm -v $PWD:/work jet-surrogate predict --hepmc /work/events.hepmc --out /work/results
#   docker run --rm -p 8080:8080 -v $PWD/service_data:/data -e JS_SERVICE_DIR=/data jet-surrogate serve
FROM ghcr.io/prefix-dev/pixi:0.63.2 AS build
WORKDIR /app
COPY pixi.toml pixi.lock pyproject.toml ./
COPY src ./src
COPY analyses ./analyses
RUN pixi install -e infer --locked && pixi shell-hook -e infer > /app/activate.sh

FROM debian:bookworm-slim
COPY --from=build /app /app
WORKDIR /app
ENTRYPOINT ["/bin/bash", "-c", "source /app/activate.sh && exec jet-surrogate \"$@\"", "--"]
CMD ["--help"]
