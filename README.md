# WWW2026-RMBRec
code of RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors
## Requirements
python==3.9.23
torch==2.3.1+cu121
numba==0.60.0
numpy==1.26.3
pandas==2.3.2
## An example to run RMBRec
### Taobao
python main.py --data_name='taobao' --irm_mode='rex' --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=1 --temperature=0.07 
### beibei
python main.py --data_name='beibei' --irm_mode='rex' --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=0.8 --temperature=0.07
### Tmall
python main.py --data_name='tmall' --irm_mode='rex' --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=1 --temperature=0.07
