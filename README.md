# AI IncidentOps — AI-Powered IT Incident Triage & MLOps Platform

An end-to-end AI and MLOps platform designed to automate IT incident management through machine learning-driven classification, priority prediction, resolution time estimation, historical incident retrieval, and actionable remediation recommendations.

The platform integrates Azure Machine Learning, MLflow, FastAPI, Docker, and Azure Kubernetes Service (AKS) to establish a scalable ML lifecycle, from model development and experimentation to containerized deployment, inference, and production monitoring.

---

## Architecture & Technology Stack

| Component | Technologies |
|---|---|
| Machine Learning | Python, Scikit-learn, TF-IDF, Random Forest |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | MLflow, DagsHub |
| ML Platform | Azure Machine Learning |
| Model Training | Azure ML Jobs & Training Pipelines |
| Backend & Inference | FastAPI, Uvicorn, Pydantic |
| API Monitoring | Prometheus Client |
| Containerization | Docker |
| Container Registry | Azure Container Registry (ACR) |
| Orchestration | Azure Kubernetes Service (AKS) |
| CI/CD | GitHub Actions |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Cloud Infrastructure | Microsoft Azure |

---

## Key Features & Capabilities

### 1. Intelligent Incident Classification
- Automatically classifies incoming IT incidents into categories such as Network, Database, Security, Storage, Performance, and Infrastructure.
- Uses machine learning models to generate category predictions with associated confidence scores.
- Applies TF-IDF feature extraction and Random Forest classification.

### 2. Incident Priority Prediction
- Predicts incident priority levels based on historical incident patterns and contextual information.
- Helps operations teams identify and prioritize critical incidents for faster resolution.

### 3. Resolution Time Estimation
- Estimates incident resolution time in minutes using historical incident data.
- Supports workload planning and more effective incident management.

### 4. Historical Incident Similarity Search
- Implements TF-IDF vectorization and cosine similarity to identify the most relevant historical incidents.
- Retrieves top matching incidents to provide additional context for troubleshooting and root-cause investigation.

### 5. Intelligent Runbook Recommendations
- Surfaces relevant historical resolution notes and category-specific remediation guidance.
- Helps engineers identify previously successful troubleshooting approaches and accelerate incident resolution.

### 6. Production Monitoring & Observability
- Exposes Prometheus metrics, including:
  - `model_predictions_total`
  - `model_prediction_confidence_score`
- Implements Kubernetes liveness and readiness probes.
- Provides health-check endpoints for application availability and deployment validation.
- Establishes monitoring foundations for tracking model inference behavior and identifying potential data drift.

---

## MLOps & Deployment Architecture

The platform follows an end-to-end MLOps workflow covering model development, experiment tracking, deployment automation, and production observability.

**Model Development & Training**
- Develops and trains machine learning models using Python and Scikit-learn.
- Uses Azure Machine Learning for managed training jobs and ML workflow execution.
- Tracks experiments, parameters, metrics, and model artifacts using MLflow and DagsHub.

**Containerization & Registry**
- Packages the FastAPI inference application into Docker containers.
- Uses a non-root production container configuration to improve runtime security.
- Stores and manages container images in Azure Container Registry.

**Kubernetes Deployment**
- Deploys containerized inference services to Azure Kubernetes Service.
- Configures Kubernetes Deployment and Service resources.
- Implements liveness and readiness probes for application health validation.
- Uses rolling updates to support controlled application releases with minimal service disruption.

**CI/CD Automation**
- Uses GitHub Actions to automate container build and deployment workflows.
- Integrates Azure Container Registry and AKS into the deployment pipeline.
- Automates build verification and application delivery to the Kubernetes environment.

---

## Repository Structure

```text
ItIncidentManagement/
│
├── src/                      # FastAPI application and core modules
│
├── ml/                       # ML training and preprocessing pipelines
│
├── k8s/                      # Kubernetes deployment and service manifests
│
├── .github/
│   └── workflows/            # GitHub Actions CI/CD pipelines
│
├── backend-fastapi/          # Lightweight FastAPI inference service
│
├── Dockerfile                # Production Docker image configuration
│
└── environment.yml            # Python environment dependencies
```

---

## Getting Started

### Prerequisites

Ensure the following tools are installed:

- Python 3.10+
- Git
- pip
- Docker (for containerized execution)
- Azure CLI (for Azure resource management, if required)

### 1. Clone the Repository

```bash
git clone https://github.com/ShravaniSU/ItIncidentManagement.git

cd ItIncidentManagement
```

### 2. Install Dependencies

```bash
pip install -r backend-fastapi/requirements.txt
```

### 3. Start the FastAPI Application

```bash
uvicorn main:app --app-dir backend-fastapi --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify the Application

Once the application starts, access the following endpoints:

| Endpoint | Description |
|---|---|
| `http://localhost:8000/docs` | Interactive Swagger API documentation |
| `http://localhost:8000/health/live` | Kubernetes liveness probe |
| `http://localhost:8000/health/ready` | Kubernetes readiness probe |
| `http://localhost:8000/metrics` | Prometheus application metrics |

---

## Live Demo

**Application:** https://it-incident-management-ten.vercel.app/

The platform's primary ML and infrastructure architecture leverages Azure Machine Learning, Azure Container Registry, and Azure Kubernetes Service.

For the publicly accessible demonstration, the React frontend is hosted on Vercel and the FastAPI backend is hosted on Render. These hosting services provide a lightweight environment for showcasing the application's functionality independently of the primary Azure-based MLOps architecture.

---

## Project Objectives

AI IncidentOps demonstrates the practical application of machine learning and MLOps principles to IT service management, bringing together:

- End-to-end machine learning workflows.
- Experiment tracking and model lifecycle management.
- Automated incident classification and prioritization.
- Containerized model inference services.
- Kubernetes-based application orchestration.
- Automated CI/CD and deployment workflows.
- Production monitoring and observability.

**The goal is to bridge the gap between machine learning development and production engineering by delivering an intelligent, scalable, and operationally focused incident management platform.**