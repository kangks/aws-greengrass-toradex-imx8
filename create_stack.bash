# BASH commands (replace exports with your AWSCLI profile, region, and S3 bucket settings)
# AWS_PROFILE contains permissions to fully create and launch the CloudFormation package and template
export AWS_PROFILE=default
export AWS_REGION=us-east-1

# S3 bucket name without s3:// or https://
export CFN_S3_BUCKET=

# S3 model URI, which requires either S3:// or https://
export ML_S3_BUCKET_URI=s3://<bucket>/torizon_pasta_model.zip

# Create a AWS IoT Thing with name that ends with _Core, such as Toradex2_Core. 
# Take note of the _Core because the Cloudformation will append _Core to the THINGNAME
# Enter the ThingName below. In this example, the Thing Name is Toradex2
export THINGNAME=toradex
export CERTIFICATE_ID=<cert ID>
                                             
# Clean up any previously created files
rm *-OUTPUT.yaml
aws cloudformation package \
--template-file cfn/mli_accelerator_dlr_models-INPUT.cfn.yaml \
--output-template-file mli_accelerator_dlr_models-OUTPUT.yaml \
--s3-bucket ${CFN_S3_BUCKET} --profile ${AWS_PROFILE} --region ${AWS_REGION}
  
# If using the AWS Console, upload the mli_accelerator_s3_models-OUTPUT.yaml and continue with the parameters.
# Below are the steps to deploy via the command line.
  
# To deploy back-end stack from CLI (change --stack-name and --parameter-overrides to expected values)
aws cloudformation deploy \
  --profile ${AWS_PROFILE} \
  --region ${AWS_REGION} \
  --stack-name greengrass-mli-accelerator-dlr \
  --template mli_accelerator_dlr_models-OUTPUT.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    CoreName=${THINGNAME} \
    CertIdParam=${CERTIFICATE_ID} \
    ModelS3Uri=${ML_S3_BUCKET_URI}

export GreengrassGroup=$(aws greengrass list-groups --query "Groups[?Name=='${THINGNAME}'].Id" --output text)

aws \
  greengrass create-deployment \
  --group-id ${GreengrassGroup} \
  --deployment-type NewDeployment \
  --group-version-id $(aws greengrass list-group-versions --group-id ${GreengrassGroup} --query "sort_by(Versions, &CreationTimestamp)[-1].Version" --output text)