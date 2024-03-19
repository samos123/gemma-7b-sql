from nemo.export import TensorRTLLM
import sys

# path to .nemo checkpoint file
nemo_model_path = sys.argv[1]
trt_model_path = sys.argv[2]

trt_llm_exporter = TensorRTLLM(model_dir=trt_model_path)
trt_llm_exporter.export(nemo_checkpoint_path=nemo_model_path, model_type="gemma", n_gpus=1)