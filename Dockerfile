# Inference image for the reinterpretation service: HepMC in, SR efficiency out.
#   docker build -t jet-surrogate .
#   docker run --rm -v $PWD:/work jet-surrogate predict --hepmc /work/events.hepmc --out /work/results
FROM ghcr.io/prefix-dev/pixi:0.63.2 AS build
WORKDIR /app
COPY pixi.toml pixi.lock pyproject.toml ./
COPY src ./src
COPY models/release ./models/release
RUN pixi install -e infer --locked && pixi shell-hook -e infer > /app/activate.sh

FROM debian:bookworm-slim
COPY --from=build /app /app
WORKDIR /app
ENTRYPOINT ["/bin/bash", "-c", "source /app/activate.sh && exec jet-surrogate \"$@\"", "--"]
CMD ["--help"]
