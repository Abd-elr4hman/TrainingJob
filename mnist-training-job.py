import os
import sagemaker
from sagemaker import get_execution_role

from sagemaker.tensorflow import TensorFlow

EXECUTION_ROLE = "arn:aws:iam::401082536487:role/AWS-SAGEMAKER-EXECUTION"
TRAIN_DATA = "s3://training-data-sagemaker-tensorflow-mnist/processing_output"

def main():
    mnist_estimator = TensorFlow(
        entry_point="mnist-training.py",
        role=EXECUTION_ROLE,
        instance_count=1,
        # instance_type="ml.m5.large",
        instance_type="local",
        framework_version="2.7",
        py_version="py38",
        source_dir= "src",
        output_path= "s3://training-data-sagemaker-tensorflow-mnist/training_output"
    )

    mnist_estimator.fit({"train":TRAIN_DATA})

if __name__ == "__main__":
    main()

