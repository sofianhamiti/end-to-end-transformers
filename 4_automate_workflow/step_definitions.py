import sagemaker
from stepfunctions.steps.choice_rule import ChoiceRule
from stepfunctions.steps.states import Task, Choice, Catch, Chain, Fail, Succeed, Wait

role = sagemaker.get_execution_role()

# ========================================================
# ================== BUILT-IN BLOCKS =====================
# ========================================================
succeed = Succeed(
    state_id="Succeed",
)

failed = Fail(
    state_id="Failed",
)

wait = Wait(
    state_id="Wait",
    seconds=60
)
# ========================================================
# =================== LAMBDA STEPS =======================
# ========================================================
initialise_execution = Task(
    state_id="InitialiseExecution",
    resource="<<<replace-with-your-lambda-arn>>>",
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

sagemaker_processing_job = Task(
    state_id="SageMakerProcessingJob",
    resource="<<<replace-with-your-lambda-arn>>>",
    result_path="$.processing.processing_job"
)


get_processing_status = Task(
    state_id="GetProcessingStatus",
    resource="<<<replace-with-your-lambda-arn>>>",
    result_path="$.processing.processing_job"
)
# ========================================================
# ================ SAGEMAKER TRAINING JOB ================
# ========================================================
sagemaker_training_job = Task(
    state_id="SageMakerTrainingJob",
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
# ========================================================
# ====== CHOSE NEXT STEP BASED ON PROCESSING STATUS ======
# ========================================================
is_processing_complete = Choice(
    state_id="IsProcessingComplete"
)
is_processing_complete.add_choice(
    ChoiceRule.StringEquals(
        variable="$.processing.processing_job.ProcessingJobStatus",
        value='InProgress'
    ),
    next_step=wait
)
is_processing_complete.add_choice(
    ChoiceRule.StringEquals(
        variable="$.processing.processing_job.ProcessingJobStatus",
        value='Failed'
    ),
    next_step=failed
)
is_processing_complete.add_choice(
    ChoiceRule.StringEquals(
        variable="$.processing.processing_job.ProcessingJobStatus",
        value='Completed'
    ),
    next_step=sagemaker_training_job
)