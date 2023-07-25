import os
import subprocess
pathdir = "/local-data/e91868xs/cifar-10-images/cifar10/cifar10/test"
model_path="./quantized_lenet_model.tflite"
for d in os.listdir(pathdir):
    d = os.path.join(pathdir, d)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            f = os.path.join(d, f)
#             print(f)
            if os.path.isfile(f):
                if os.path.splitext(f)[-1][1:] == "png":
                    print(f)
                    parts = f.split("/")                    
                    category = parts[-2]
                    command = 'python ./tflite_tensor_outputter/tflite_tensor_outputter_cifar10.py --image ' + f + ' --model_file '+ model_path + ' --label_file ./tflite_tensor_outputter/imagenet-labels.txt ' + '--output_dir /local-data/e91868xs/lenet_output/'+ category + "/" + os.path.split(f)[1] +"/"
                    print(command)
                    subprocess.call(command, shell=True)
