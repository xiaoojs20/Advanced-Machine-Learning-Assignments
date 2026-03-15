ssh xiaojs24@h12

cd ../../flash2/aml/xiaojs24/AML_1/src

tmux new -s train_glm4

tmux attach -t train_glm4

conda activate xiaojs24

export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
conda activate xiaojs24

python run_glm4.py
python chatglm4.py