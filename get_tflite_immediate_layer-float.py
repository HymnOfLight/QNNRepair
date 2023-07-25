import os
import subprocess
pathdir = "../imagemini-1000/imagenet-mini/val"
model_path="./model/mobilenet_v2_1.0_224.tflite"
for d in os.listdir(pathdir):
    d = os.path.join(pathdir, d)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            f = os.path.join(d, f)
            print(f)
            if os.path.isfile(f):
                if os.path.splitext(f)[-1][1:] == "JPEG":
                    command = 'python ./tflite_tensor_outputter/tflite_tensor_outputter.py --image ' + f + ' --model_file '+ model_path + ' --label_file ./tflite_tensor_outputter/imagenet-labels.txt ' + '--output_dir ./output/' + os.path.split(f)[1] +"/"
                    print(command)
                    subprocess.call(command, shell=True)
