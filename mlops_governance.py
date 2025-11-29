import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import random

# --- 1. STRICT DATA & PARAMETER VALIDATION (The "Define" Phase) ---
# Enforcing a schema on the training data and parameters to ensure quality and compliance.

class TrainingDataSchema(BaseModel):
    """Defines the expected schema for the training dataset."""
    feature_a: List[float]
    feature_b: List[float]
    target_label: List[int]
    
    @validator('feature_a', 'feature_b', 'target_label')
    def lists_must_be_non_empty(cls, v):
        if not v:
            raise ValueError('Data lists must not be empty.')
        return v

class TrainingParams(BaseModel):
    """Defines the hyper-parameters used for the training run."""
    model_type: str = Field(..., description="e.g., RandomForest, XGBoost")
    n_estimators: int
    learning_rate: float

# --- 2. RUN AND ARTIFACT TRACKING (The "Measure" Phase) ---
# A simulation of the governance payload that would be logged to an MLflow/Registry server.

class ModelRunArtifact(BaseModel):
    """The central governance object logged for every training run."""
    run_id: str
    timestamp: datetime.datetime
    model_path: str = Field(..., description="S3/GS/Artifact path where the model is stored.")
    git_commit: str
    data_version_tag: str
    metrics: Dict[str, float]
    params: TrainingParams
    lineage_verified: bool = False
    
def simulate_training_run(params: TrainingParams, data_schema: TrainingDataSchema) -> ModelRunArtifact:
    """
    Simulates a training and logging process. In a real system, this calls MLflow.
    """
    
    # 1. Validation Check before training starts
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] INFO: Validating input data and parameters...")
    # This step uses the Pydantic schemas to validate data consistency
    
    # 2. Simulate Training and Metrics Logging
    accuracy = random.uniform(0.75, 0.95)
    f1_score = random.uniform(0.70, 0.90)
    
    # 3. Create the artifact record
    artifact = ModelRunArtifact(
        run_id=f"run-{random.randint(1000, 9999)}",
        timestamp=datetime.datetime.now(),
        model_path=f"s3://model-registry/prod/model_v_{random.randint(1, 100)}.pkl",
        git_commit="a4f3d2c1",
        data_version_tag="v20251129-cleaned",
        metrics={"accuracy": accuracy, "f1_score": f1_score},
        params=params,
    )
    
    # Log the complete artifact to the registry (conceptual step)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SUCCESS: Run tracked. Run ID: {artifact.run_id}")
    return artifact

# --- 3. AUTOMATED MODEL VALIDATION (The "Analyze" Phase) ---

def automated_validation_gate(artifact: ModelRunArtifact, min_accuracy: float = 0.85) -> bool:
    """Checks if the model meets minimum performance and compliance thresholds."""
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] INFO: Running automated validation gate...")
    
    if artifact.metrics.get("accuracy", 0) < min_accuracy:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] FAIL: Accuracy ({artifact.metrics['accuracy']:.2f}) below threshold ({min_accuracy}).")
        return False
    
    if not artifact.lineage_verified:
        # In a real system, this checks if all data/code sources were logged correctly.
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] FAIL: Lineage verification is missing (Compliance Failure).")
        return False
        
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] PASS: Model meets performance and compliance requirements.")
    return True

# --- 4. PROMOTION AND METADATA LOGGING (The "Control" Phase) ---

class DeploymentRecord(BaseModel):
    """The final auditable record for a production deployment."""
    run_id: str
    promotion_timestamp: datetime.datetime
    approved_by: str
    target_environment: str
    deployment_tool: str = "GitHub Actions"
    
def promote_to_production_registry(artifact: ModelRunArtifact, approver: str):
    """
    Simulates promoting the model and logging the final deployment metadata.
    """
    if not automated_validation_gate(artifact):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ABORT: Cannot promote Run {artifact.run_id} due to validation failure.")
        return

    # Assuming the lineage was verified successfully for a real production model
    artifact.lineage_verified = True 
    
    record = DeploymentRecord(
        run_id=artifact.run_id,
        promotion_timestamp=datetime.datetime.now(),
        approved_by=approver,
        target_environment="Production"
    )
    
    # Log the final deployment record to the audit database (conceptual step)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] PRODUCTION: Model {artifact.run_id} promoted to {record.target_environment} by {record.approved_by}.")
    print(f"--- Full Deployment Audit Record ---")
    print(record.model_dump_json(indent=2))


# --- EXECUTION ---

if __name__ == "__main__":
    
    # 1. Define Training Parameters and Simulate Data
    
    # Example 1: A successful run that meets all governance requirements
    successful_params = TrainingParams(model_type="RandomForest", n_estimators=100, learning_rate=0.01)
    successful_data = TrainingDataSchema(feature_a=[1.2, 2.5], feature_b=[5.1, 4.3], target_label=[0, 1])
    
    print("\n\n=== GOVERNANCE RUN 1: Successful Model Promotion ===")
    
    try:
        # Simulate training and tracking
        successful_run = simulate_training_run(successful_params, successful_data)
        
        # Simulate promotion after successful validation
        # The line below will promote only if the simulated accuracy is >= 0.85
        promote_to_production_registry(successful_run, approver="Sungroo (VP/Director)")

    except Exception as e:
        print(f"FATAL ERROR: {e}")

    # Example 2: A simulated run that fails validation (e.g., poor accuracy)
    
    # Setting accuracy threshold high to demonstrate failure
    MIN_PROD_ACCURACY = 0.98 
    
    print(f"\n\n=== GOVERNANCE RUN 2: Validation Failure (Accuracy < {MIN_PROD_ACCURACY}) ===")
    
    try:
        failing_params = TrainingParams(model_type="XGBoost", n_estimators=50, learning_rate=0.1)
        failing_data = TrainingDataSchema(feature_a=[0.5, 0.5], feature_b=[0.5, 0.5], target_label=[0, 0])
        
        # Simulate training and tracking (will generate a random, likely low score)
        failing_run = simulate_training_run(failing_params, failing_data)
        
        # Attempt to promote
        promote_to_production_registry(failing_run, approver="Sungroo (VP/Director)")

    except Exception as e:
        print(f"FATAL ERROR: {e}")
