######################### 
#     Skip-Reduce       #
#########################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name deviceDropping_X2 --project experiments_ready



#########################
#    Random Pruning     #
#########################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/50/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name random_50 --project experiments_ready

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/75/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name random_75 --project experiments_ready

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/90/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name random_90 --project experiments_ready


#########################
#        ADAPTIVE       #
#                       #
#########################

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name inorder --project adaptive

# export LD_LIBRARY_PATH=//src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume


# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 600 --batch 256 --data VOC.yaml --weights '' --cfg yolov5m.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name hybrid --project adaptive

# export LD_LIBRARY_PATH=//src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# python3 -m torch.distributed.run --nproc_per_node 4 train.py --resume

#########################
#    KERNEL PROFILE     #
#                       #
#########################

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/YOLO%p.csv python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 10 --batch 256 --data VOC.yaml --weights '' --cfg yolov5s.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name profile --project profile


# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/YOLO_X1%p.csv python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 10 --batch 256 --data VOC.yaml --weights '' --cfg yolov5s.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name profile --project profile


# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/YOLO_X2%p.csv python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 10 --batch 256 --data VOC.yaml --weights '' --cfg yolov5s.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name profile --project profile


# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/YOLO_X1Y1%p.csv python3 -m torch.distributed.run --nproc_per_node 4 train.py --epochs 10 --batch 256 --data VOC.yaml --weights '' --cfg yolov5s.yaml --hyp ./data/hyps/hyp.VOC.yaml --device 0,1,2,3 --cache ram --name profile --project profile


# &&&&&&&&& # 
# commands are not 100% accurate, e.g. for adaptive i used to manually interrupt the training, change comm lib, and resume, and etc. For kernel profiling, I manually set the num of iters.
