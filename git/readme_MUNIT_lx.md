Q:
    1,trainer.gen_a.load_state_dict(state_dict['a'])
    KeyError: 'a'
        A: 模型使用gen_***.pt,使用optimizer.pt会报错，dis用于判别？
        
    2,No module named 'torch.utils.serialization'
        A：was removed   使用torch0.4版本
        
pytorch 0.4 安装：
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl

python train.py --config configs/cat2dog.yaml --resume
训练文件夹里面只能有4个，不能有其他。

python test.py --config configs/cityscapes.yaml --input inputs/frame.png --output_folder outputs --checkpoint outputs/cityscapes/checkpoints/optimizer.pt --a2b 1 --style inputs/guide_scapes.jpg  #a2b 0/1
