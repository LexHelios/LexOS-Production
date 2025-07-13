## Running This Project with Docker

This project is containerized using Docker and Docker Compose for streamlined deployment and development. Below are the project-specific instructions and requirements for running the application in a Dockerized environment.

### Project-Specific Docker Requirements
- **Base Image:** `python:3.11-slim` (Python 3.11)
- **System Dependencies:** Installed in the image: `build-essential`, `gcc`, `g++`, `python3-dev`, `libpq-dev`, `libmagic1`, `git`, `curl`
- **Python Dependencies:** Installed from `requirements.txt` into a virtual environment (`.venv`) during the build process.
- **User:** Runs as a non-root user `lexos` for improved security.

### Environment Variables
- No required environment variables are specified in the Dockerfile or Compose file by default.
- If you need to set environment variables, you can uncomment and use the `env_file: ./.env` line in `docker-compose.vllm.yml`.

### Build and Run Instructions
1. **Build and Start the Application:**
   ```sh
   docker compose -f docker-compose.vllm.yml up --build
   ```
   This command builds the Docker image using `Dockerfile.production` and starts the service defined in `docker-compose.vllm.yml`.

2. **Stopping the Application:**
   ```sh
   docker compose -f docker-compose.vllm.yml down
   ```

### Special Configuration
- The application uses a custom entrypoint: `startup_wrapper.py`.
- Health checks are configured to monitor the main app endpoint at `http://localhost:8080/health`.
- The image creates and uses the following directories: `/app/logs`, `/app/lexos_memory`, `/app/uploads`, `/app/static` (all owned by the `lexos` user).
- No external services (databases, caches) are configured by default.
- No persistent volumes are set up by default. If you need data persistence, add volumes to the Compose file as needed.

### Exposed Ports
- **8080:** Main application port (mapped to host 8080)
- **8081:** Secondary port (mapped to host 8081)

Both ports are exposed and mapped in the Compose file. Access the main application at `http://localhost:8080/` after starting the service.

---

_If you add new services (e.g., databases) or environment variables, update the Docker Compose file and this section accordingly._