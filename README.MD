# Running Pasta classification on Toradex iMX8 using DLR model on AWS IoT Greengrass

## Overview on how to setup

There are 2 main components in setting up the pasta classification, the AWS side and the edge side on the iMX8.

1. The AWS side involves creating an IoT Thing and use that as the GreenGrass Core of a GreenGrass Group
2. The edge side on the iMX8 involves building the Dockerfile, download the certs and config created from the AWS side, and runs the Greengrass in a Docker container

## AWS setup steps

### 1. Create an AWS IoT Certificate
1. Run the command to generate a certificate
```
aws iot create-keys-and-certificate --certificate-pem-outfile "toradex.cert.pem" --public-key-outfile "toradex.public.key" --private-key-outfile "toradex.private.key" --set-as-active --query certificateId --output text
```
2. Take note of the `certificate ID` which will be needed to create the Greengrass Core

### 2. Upload the trained DLR model
1. Download the folder `model/` from https://github.com/toradex/aws-nxp-ai-at-the-edge/tree/master/container_inference/model
2. Zip the contents in the `model/` as a single zip file, such as 
```
cd model/ && zip pasta-model.zip ./
```
3. Upload the `pasta-model.zip` to a S3 bucket, such as `<AWS ACCOUNT ID>-<AWS REGION>-greengrass-models`

### 3. Create a AWS IoT Greengrass Group
1. Edit the file `create_stack.bash` and update the configuration
* AWS_PROFILE - the AWS Profile that CLI should use to create the Cloudformation stack
* AWS_REGION - the region that the stack should be created in
* CFN_S3_BUCKET - your S3 bucket that CloudFormation can export the transformed template to. Example bucket name can be `<AWS ACCOUNT ID>-<AWS REGION>-cloudformation-templates`
* ML_S3_BUCKET_URI - the S3 bucket URI that you uploaded the `pasta-model.zip`. In this example, it was s3://`<AWS ACCOUNT ID>-<AWS REGION>-greengrass-models`/pasta-model.zip
* THINGNAME - Name for the GreenGrass Core without _Core, such as Toradex.
* CERTIFICATE_ID - The certificate ID that you created from step 1
2. Run the `create_stack.bash`
3. If the stack creation failed, check the events in the AWS Console https://console.aws.amazon.com/cloudformation/home?#/stacks/events

### 4. Inspect the stack
1. Go to the Greengrass Group in the AWS Console https://console.aws.amazon.com/iot/home?#/greengrass/grouphub
2. Select the Greengrass Group that was created, such as `Toradex`
3. Select `Resource` from the left hand side
4. Select `Machine Learning` from the tabs
5. Select the 3 dots "..." of the `mlModel`, select "Edit resource"
6. Make sure in `Identify resource owner and set access permissions`, `Specify OS group and permissions` is selected. If not, select `Specify OS group and permissions`, and enter `998` as the `OS group ID`

## iMX8 setup steps

### 1. Build the Dockerfile
1. scp the contents of `docker/` folder to iMX8, such as
```
scp -r docker/* torizon@<iMX8 IP>:~/greengrass-docker/
```
2. ssh to torizon and build the docker file
```
$ cd greengrass-docker
$ docker-compose build
```

### 2. Create config and certs folder and upload the configuration
1. In the iMX8 `~/greengrass-docker/`, make 4 folders for the AWS IoT Greengrass container to mount as external volumes:
   * ~/greengrass-docker/certs
   * ~/greengrass-docker/config
   * ~/greengrass-docker/deployment
   * ~/greengrass-docker/log
1. scp the certificates created in [create an aws iot certificate](#1-create-an-aws-iot-certificate) to the iMX8 `~/greengrass-docker/certs` folder
2. Create a Greengrass configuration by changing the values from the template below:

```
{
  "coreThing" : {
    "caPath" : "root.ca.pem",
    "certPath" : "toradex.cert.pem",
    "keyPath" : "toradex.private.key",
    "thingArn" : "arn:aws:iot:<AWS REGION>:<AWS ACCOUNT ID>:thing/<Thing Name, with _Core, such as Toradex_Core>",
    "iotHost" : "<Your AWS IOT Endpoint address",
    "ggHost" : "greengrass-ats.iot.us-east-1.amazonaws.com",
    "keepAlive" : 600
  },
  "runtime" : {
    "cgroup" : {
      "useSystemd" : "yes"
    }
  },
  "managedRespawn" : false,
  "crypto" : {
    "principals" : {
      "SecretsManager" : {
        "privateKeyPath" : "file:///greengrass/certs/toradex.private.key"
      },
      "IoTCertificate" : {
        "privateKeyPath" : "file:///greengrass/certs/toradex.private.key",
        "certificatePath" : "file:///greengrass/certs/toradex.cert.pem"
      }
    },
    "caPath" : "file:///greengrass/certs/root.ca.pem"
  }
}
```

### 3. Run the container
1. In the `~/greengrass-docker/` folder, run `docker-compose up`. If the docker image was successfully build, you should be able to see the message as below:

```
$ docker-compose up
Starting aws-iot-greengrass-latest ... done
Attaching to aws-iot-greengrass-latest
aws-iot-greengrass-latest | grep: /greengrass/ggc/deployment/group/group.json: No such file or directory
aws-iot-greengrass-latest | Process with pid 14 does not exist already
aws-iot-greengrass-latest | Setting up greengrass daemon
aws-iot-greengrass-latest | Validating hardlink/softlink protection
aws-iot-greengrass-latest | Waiting for up to 1m10s for Daemon to start
aws-iot-greengrass-latest | 
aws-iot-greengrass-latest | Greengrass successfully started with PID: 13
```

2. You can monitor the Greengrass log by tailing the log in ``~/greengrass-docker/logs`, such as

```
tail -f ~/greengrass-docker/log/system/runtime.log
```

If Greengrass Core started successfully, you should see this messages in runtime.log

```
# tail -F log/system/runtime.log
[2020-02-08T05:52:25.054Z][DEBUG]-Subscribe retry configuration.	{"IntervalInSeconds": 60}
[2020-02-08T05:52:25.054Z][INFO]-Start subscribing.	{"numOfTopics": 2, "clientId": "toradex_Core"}
[2020-02-08T05:52:25.054Z][INFO]-Trying to subscribe to topic $aws/things/toradex_Core-gda/shadow/update/delta
[2020-02-08T05:52:25.054Z][DEBUG]-Subscribe	{"topic": "$aws/things/toradex_Core-gda/shadow/update/delta", "qos": 0}
[2020-02-08T05:52:25.171Z][DEBUG]-Subscribed to topic.	{"topic": "$aws/things/toradex_Core-gda/shadow/update/delta"}
[2020-02-08T05:52:25.171Z][INFO]-Trying to subscribe to topic $aws/things/toradex_Core-gda/shadow/get/accepted
[2020-02-08T05:52:25.171Z][DEBUG]-Subscribe	{"topic": "$aws/things/toradex_Core-gda/shadow/get/accepted", "qos": 0}
[2020-02-08T05:52:25.295Z][DEBUG]-Subscribed to topic.	{"topic": "$aws/things/toradex_Core-gda/shadow/get/accepted"}
[2020-02-08T05:52:25.295Z][DEBUG]-Publish	{"topic": "$aws/things/toradex_Core-gda/shadow/get", "qos": 1}
[2020-02-08T05:52:25.403Z][INFO]-All topics subscribed.	{"clientId": "toradex_Core"}
```

3. Deploy the Greengrass from AWS Console. If this is your first time deploying the group, select `Automatic detection` in the page `Configure how Devices discover your Core`

4. If deployment was successful, you can restart the UI to connect to the inference.py running in the Greengrass using command
```
docker restart local-ui
```
