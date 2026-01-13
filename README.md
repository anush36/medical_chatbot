# Local LLM Chatbot MVP

A flexible chatbot MVP that supports local models, OpenAI, and **self-hosted MedGemma on Google Cloud**.

## Features

- **Local Inference:** Run small models (Phi-2) on your own hardware.
- **OpenAI Support:** Connect to GPT-3.5/4.
- **MedGemma (Medical AI):** Deploy a private, serverless instance of Google's MedGemma 4B on Cloud Run with GPUs.
- **Cost Efficient:** The Cloud Run setup scales to zero (costs $0 when not in use).

---

## Installation

```bash
poetry install
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to configure your preferred model provider.

### Option A: Local Model (Default)

Runs on your laptop CPU/GPU.

```env
MODEL_PROVIDER=local
LOCAL_MODEL_NAME=microsoft/phi-2
LOCAL_MAX_TOKENS=64
```

### Option B: OpenAI API

Connects to OpenAI's public API.

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### Option C: MedGemma (Google Cloud)

Connects to your private vLLM instance on Cloud Run.

```env
MODEL_PROVIDER=medgemma
# You will get this URL after running the Terraform deployment steps below
MEDGEMMA_BASE_URL=https://medgemma-service-xyz.run.app/v1
MEDGEMMA_API_KEY=fake-key
MEDGEMMA_MODEL=google/medgemma-4b-it
```

---

## Deployment Guide: MedGemma on GCP

If you want to run the medical model, you need to deploy the infrastructure first.

### Prerequisites

- Google Cloud Account (Billing enabled).
- Terraform installed.
- gcloud CLI installed and authenticated (`gcloud auth application-default login`).
- Hugging Face Token (Read access) and acceptance of the MedGemma 4B License.

### Step 1: Deploy Infrastructure

We use Terraform to spin up a secured, serverless GPU container.

```bash
cd gcp_infra

# 1. Initialize Terraform
terraform init

# 2. Deploy
terraform apply
```

You will be asked for:

- `project_id`: Your GCP Project ID.
- `hf_token`: Your Hugging Face token (invisible when typing).

### Step 2: Connect the App

Once Terraform finishes, it will output an API URL (e.g., `https://medgemma-service-....run.app`).

- Copy this URL.
- Open your `.env` file.
- Set `MODEL_PROVIDER=medgemma`.
- Paste the URL into `MEDGEMMA_BASE_URL`.

---

## Running the Application

### Backend (FastAPI)

```bash
poetry run uvicorn backend.main:app --reload
```

The backend runs at [http://localhost:8000](http://localhost:8000).

### Frontend (Streamlit)

In a new terminal:

```bash
poetry run streamlit run frontend/app.py
```

The UI will open at [http://localhost:8501](http://localhost:8501).

---

## ⚠️ Important: Cost & Teardown

The MedGemma setup uses an NVIDIA L4 GPU. While it is configured to "Scale to Zero" (stopping costs when idle), you should destroy the infrastructure if you are done with the project to prevent any accidental charges.

To Destroy the Infrastructure:

```bash
cd gcp_infra
terraform destroy
```

Type `yes` to confirm. This removes the GPU service, secrets, and permissions.

---

## Architecture

- **Frontend:** Streamlit (Agnostic to backend model).
- **Backend:** FastAPI with Strategy Pattern for model switching.

**Providers:**

- `LocalModelProvider`: HuggingFace Transformers.
- `OpenAIModelProvider`: Works for both OpenAI and vLLM (MedGemma).

**Infrastructure:**

- **Compute:** Google Cloud Run (Serverless Container).
- **Hardware:** NVIDIA L4 GPU.
- **Model Server:** vLLM (High-performance inference engine).
- **Security:** Secrets stored in Google Secret Manager; Service runs with minimal IAM permissions.