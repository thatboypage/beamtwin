# BeamTwin web app: a recipe for a self-contained box ("image") that runs the app anywhere.
#
# Build:  docker build -t beamtwin .
# Run:    docker run -p 8000:8000 beamtwin      then open http://localhost:8000
#
# The image holds only what the web app needs: NumPy, SciPy, FastAPI and the small
# exported model. No PyTorch, no training data, no notebooks.

# 1. Start from a small official image that already has Python installed.
FROM python:3.11-slim

# 2. Sensible Python settings for a server: no .pyc files, print logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 3. Work inside /app in the container.
WORKDIR /app

# 4. Install the package and its web dependencies first. Docker caches each step,
#    so later code-only changes rebuild quickly without reinstalling libraries.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[web]"

# 5. Copy the web API, the web page and the trained model.
COPY api ./api
COPY app ./app
COPY models/surrogate_hybrid.npz ./models/surrogate_hybrid.npz
ENV BEAMTWIN_MODEL=/app/models/surrogate_hybrid.npz

# 6. Run as an ordinary user rather than the all-powerful "root" user (safer).
RUN useradd --create-home beamtwin
USER beamtwin

# 7. Start the server. Hosting services tell the app which port to use through the
#    PORT variable; when it is not set (e.g. on your laptop) it uses 8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
