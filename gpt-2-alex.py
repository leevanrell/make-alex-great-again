#/usr/bin/python3
import gpt_2_simple as gpt2
import os

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

file_name = "./src/alex_jones.txt"

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
			  run_name='run2',
              steps=1000,
			  save_every=50,
			  print_every=5,
			  sample_every=10,
			  learning_rate = 0.0001
)

gpt2.generate_to_file(sess)