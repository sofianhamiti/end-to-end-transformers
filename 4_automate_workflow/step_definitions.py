import sagemaker
from stepfunctions.steps.states import Task, Catch, Chain, Fail, Succeed

role = sagemaker.get_execution_role()
input_folder = '/opt/ml/processing/input'
output_folder = '/opt/ml/processing/output'

# ========================================================
# ================== BUILT-IN BLOCKS =====================
# ========================================================
failed = Fail(
    state_id="Failed",
)

succeed = Succeed(
    state_id="Succeed",
)
# ========================================================
# =================== LAMBDA STEP ========================
# ========================================================
initialise_execution = Task(
    state_id="InitialiseExecution",
    resource="PLACE YOUR CREATE LAMBDA FUNCTION ARN HERE",
    parameters={
        "project.$": "$.project",
        "data.$": "$.data",
        "execution.$": "$$.Execution.Id"
    },
    result_path="$.data"
)

initialise_execution.add_catch(
    Catch(
        error_equals=["States.ALL"],
        next_step=failed,
        result_path="$"
    )
)
# ========================================================
# =============== SAGEMAKER PROCESSING JOB ===============
# ========================================================
sagemaker_processing_job = Task(
    state_id="Prepare Data",
    resource="arn:aws:states:::sagemaker:createProcessingJob.sync",
    parameters={
        "ProcessingJobName.$": "$.data.job_name",
        "AppSpecification": {
            "ContainerArguments": [
                f"--input={input_folder}",
                f"--output={output_folder}"
            ],
            "ImageUri.$": "$.processing.container"
        },
        "ProcessingInputs": [ 
            {
                "InputName": "input",
                "S3Input": {
                    "LocalPath": input_folder,
                    "S3Uri.$": "$.data.processing_input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [ 
                {
                    "OutputName": "preprocessed",
                    "S3Output": {
                        "LocalPath": output_folder,
                        "S3Uri.$": "$.data.processing_output",
                        "S3UploadMode": "EndOfJob"
                    }
                }
            ]
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType.$": "$.processing.instance_type",
                "VolumeSizeInGB": 30
            }
        },
        "RoleArn": role
    },
    result_path="$.processing.processing_job"
)

sagemaker_processing_job.add_catch(
    Catch(
        error_equals=["States.ALL"],
        next_step=failed,
        result_path="$"
    )
)
# ========================================================
# ================ SAGEMAKER TRAINING JOB ================
# ========================================================
sagemaker_training_job = Task(
    state_id="Train Model",
    resource="arn:aws:states:::sagemaker:createTrainingJob.sync",
    parameters={
        "TrainingJobName.$": "$.data.job_name",
        "AlgorithmSpecification": {
          "TrainingImage.$": "$.training.container",
          "TrainingInputMode": "File"
        },
        "InputDataConfig": [
          {
            "ChannelName": "train",
            "DataSource": {
              "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri.$": "$.data.training_input",
                "S3DataDistributionType": "FullyReplicated"
              }
            }
          }
        ],
        "ResourceConfig": {
          "InstanceCount.$": "$.training.instance_count",
          "InstanceType.$": "$.training.instance_type",
          "VolumeSizeInGB": 50
        },
        "HyperParameters": {
          "model_name.$": "$.training.hyperparameters.model_name",
          "data_folder.$": "$.training.hyperparameters.data_folder",
          "output_folder.$": "$.training.hyperparameters.output_folder",
          "epochs.$": "$.training.hyperparameters.epochs",
          "learning_rate.$": "$.training.hyperparameters.learning_rate",
          "batch_size.$": "$.training.hyperparameters.batch_size",
          "seed.$": "$.training.hyperparameters.seed",
          "max_len.$": "$.training.hyperparameters.max_len",
          "sagemaker_container_log_level.$": "$.training.hyperparameters.sagemaker_container_log_level",
          "sagemaker_enable_cloudwatch_metrics.$": "$.training.hyperparameters.sagemaker_enable_cloudwatch_metrics",
          "sagemaker_program.$": "$.training.hyperparameters.sagemaker_program",
          "sagemaker_region.$": "$.training.hyperparameters.sagemaker_region",
          "sagemaker_submit_directory.$": "$.training.hyperparameters.sagemaker_submit_directory"
        },
        "OutputDataConfig": {
          "S3OutputPath.$": "$.data.training_output"
        },
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 3600
        },
        "RoleArn": role
    },
    result_path="$.training"
)

sagemaker_training_job.add_catch(
    Catch(
        error_equals=["States.ALL"],
        next_step=failed,
        result_path="$"
    )
)