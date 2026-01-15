variable "project_id" {
  description = "Your Google Cloud Project ID"
  type        = string
}

variable "hf_token" {
  description = "Your Hugging Face Read Token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "Google Cloud Region"
  type        = string
  default     = "us-east4" # Northern Virginia (Best for L4)
}

variable "model_name" {
  description = "The HuggingFace model ID"
  type        = string
  # <--- FIXED: The ACTUAL medical model you wanted
  default     = "google/medgemma-4b-it" 
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "compute.googleapis.com",
    "secretmanager.googleapis.com",
    "billingbudgets.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# Secrets for HF Token
resource "google_secret_manager_secret" "hf_token_secret" {
  secret_id = "vllm-hf-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "hf_token_version" {
  secret      = google_secret_manager_secret.hf_token_secret.id
  secret_data = var.hf_token
}

# Service Account
resource "google_service_account" "vllm_sa" {
  account_id   = "medgemma-runner"
  display_name = "Service Account for MedGemma"
}

resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = google_secret_manager_secret.hf_token_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.vllm_sa.email}"
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "vllm_service" {
  name     = "medgemma-service"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.vllm_sa.email
    
    scaling {
      max_instance_count = 1
      min_instance_count = 0 
    }

    containers {
      # Use a recent vLLM version to ensure MedGemma support

            image = "vllm/vllm-openai:latest" 
      resources {
        limits = {
          cpu    = "4"
          memory = "16Gi" 
          "nvidia.com/gpu" = "1"
        }
      }

      args = [
        "--model", var.model_name,
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192", # MedGemma supports longer context
        "--trust-remote-code",      # Required for specialized architectures
        "--dtype", "bfloat16"       # Native precision for Gemma
      ]

      env {
        name = "HUGGING_FACE_HUB_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.hf_token_secret.secret_id
            version = "latest"
          }
        }
      }
    }
    
    node_selector = {
      "run.googleapis.com/accelerator" = "nvidia-l4"
    }
    
    annotations = {
      "run.googleapis.com/launch-stage" = "BETA"
    }
  }
  depends_on = [google_secret_manager_secret_version.hf_token_version]
}

output "api_url" {
  value = google_cloud_run_v2_service.vllm_service.uri
}

# 1. Get the email of the person running this terraform (You)
data "google_client_config" "default" {}

# 2. Allow YOU to invoke the service
resource "google_cloud_run_v2_service_iam_member" "developer_access" {
  name     = google_cloud_run_v2_service.vllm_service.name
  location = google_cloud_run_v2_service.vllm_service.location
  role     = "roles/run.invoker"
  member   = "user:${data.google_client_config.default.account}"
  
  depends_on = [google_cloud_run_v2_service.vllm_service]
}