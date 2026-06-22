
import sys
ROOT = '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI'
sys.path.insert(0, ROOT)

#  RETE E DATASET 
NETWORK_NAME = 'MOBILENETV2_X1_0'
DATASET = 'CIFAR100'

#  OPTIMIZER 
OPTIMIZER = 'adam'        # 'adam' o 'sgd'
LR = 0.0005
WEIGHT_DECAY = 5e-4

#  SCHEDULER 
SCHEDULER = 'stepLR'      
STEP_LR = 10          # ogni quante epoche scala il LR
GAMMA = 0.80           # fattore moltiplicativo: LR = LR * GAMMA

#  BATCH SIZE 
USE_BATCH_SCHEDULER = True   # True → batch size dinamico, False → fisso
BATCH_SIZE = 64
MAX_BATCH_SIZE = 256
MIN_BATCH_SIZE = 32
STEP_BATCH_SIZE = 10       # ogni quante epoche cambia
GAMMA_BATCH_SIZE = 0.9      # fattore moltiplicativo

#  TRAINING 
EPOCHS = 200
SEED = 42  #DO NOT CHANGE IT, NEEDS TO BE THE SAME FOR REPRODUCIBILITY
DEVICE = 'cuda'

#  VALIDATION
USE_VALIDATION = False
VAL_SPLIT = 0.1

# EARLY STOPPING
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0001

#  PATHS 
SAVE_DIR = f'{ROOT}/models/{DATASET}/pretrained'
DATASET_ROOT = f'{ROOT}/data'
RESULTS_DIR = f'{ROOT}/training/results/{DATASET}'

