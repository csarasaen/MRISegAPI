# main.tf — provision an EKS cluster for the MRI segmentation service

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  # Remote state so the team shares the same state file
  backend "s3" {
    bucket  = "mri-seg-terraform-state"
    key     = "prod/terraform.tfstate"
    region  = "eu-central-1"
    encrypt = true # HIPAA-adjacent: encrypt state at rest
  }
}

provider "aws" { region = var.region }

# ── VPC ───────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"

  name            = "${var.project}-vpc"
  cidr            = "10.0.0.0/16"
  azs             = ["${var.region}a", "${var.region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  # Cost optimisation: one NAT gateway instead of one per AZ.
  # Saves ~$32/month in non-production — set to false for prod HA.
  enable_nat_gateway = true
  single_nat_gateway = var.environment != "prod"

  tags = local.common_tags
}

# ── EKS cluster ───────────────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.8.5"

  cluster_name    = "${var.project}-${var.environment}"
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    gpu_nodes = {
      # GPU nodes for model inference — Spot instances cut cost ~70% vs on-demand
      instance_types = ["g4dn.xlarge"]
      capacity_type  = "SPOT"
      min_size       = 1
      max_size       = 5
      desired_size   = 2

      labels = {
        workload = "medical-inference"
      }
    }

    cpu_nodes = {
      # CPU nodes for the API layer — cheaper and always available
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 3
      desired_size   = 1
    }
  }

  tags = local.common_tags
}

locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
