The link of SA task is from https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

```bash
python run_glue.py \
--model_name_or_path google-bert/bert-base-uncased \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 384 \
--per_device_train_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 3 \
--output_dir models/bert-base-cased-sst-2-lr==1e-5_max_sl==384
```

You can download the model weights of above bash command via 
```bash
https://drive.google.com/file/d/1MjGLdc_Dy0K_B7k92xw-h5sCtjVmYIuD/view?usp=drive_link
```
