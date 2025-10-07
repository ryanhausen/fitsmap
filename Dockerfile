FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# we need git to run generate the version for the build
RUN apt update
RUN apt upgrade -y
RUN apt install -y git-all

WORKDIR /app

COPY . .

RUN uv sync --group prd

ENTRYPOINT ["uv", "run", "--group", "prd", "fitsmap"]
CMD [ "--help" ]