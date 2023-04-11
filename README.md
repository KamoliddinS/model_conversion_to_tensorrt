# model_conversion_to_tensorrt



### convert onnx model to tensorrt model

```bash
docker run -it --gpus all --rm -v ${PWD}/models/tensorrt_fp16_model/1:/trt_optimize nvcr.io/nvidia/tensorrt:22.11-py3

cd /trt_optimize
/workspace/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```



### Trition Inference Server directory structure

```bash
{PWD}/models
         |
         |-- tensorrt_fp16_model
         |          |
         |          |-- config.pbtxt
         |          |-- 1
         |          |   |
         |          |   |--model.plan

```


#### Example config.pbtxt

```bash
name: "tensorrt_fp16_model"
platform: "tensorrt_plan"
max_batch_size: 32

input [ 
    {
        name: "input_1"
        data_type: TYPE_FP16
        dims: [ 224, 224, 3 ]

    }
]

output [
    {
        name: "predictions"
        data_type: TYPE_FP16
        dims: [ 1000 ]
    }
]
```