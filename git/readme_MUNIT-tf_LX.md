test

    python main.py --phase guide --dataset cityscapes --batch_size 1 --direction a2b --guide_img guide.jpg --result_dir result --img_h 1080 --img_w 1920


train

    python main.py --phase train --dataset cityscapes --batch_size 1









##### Discriminator #####
Discriminator layer :  4
Multi-scale Dis :  3
Traceback (most recent call last):
  File "/home/xiang/venv/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 510, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "/home/xiang/venv/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1146, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/home/xiang/venv/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 983, in _TensorTensorConversionFunction
    (dtype.name, t.dtype.name, str(t)))
ValueError: Tensor conversion requested dtype string for Tensor with dtype float32: 'Tensor("arg0:0", shape=(), dtype=float32)'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 112, in <module>
    main()
  File "main.py", line 93, in main
    gan.build_model()
  File "/home/xiang/git/MUNIT-Tensorflow/MUNIT.py", line 243, in build_model
    trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        
