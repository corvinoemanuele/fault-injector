# ===================== RETE E DATASET =====================
NETWORK_NAME = 'VGG13_BN_SGD_EP_200_LR_01'
DATASET = 'CIFAR100'

# ===================== PATH =====================
ROOT = '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI'
DATASET_ROOT = f'{ROOT}/data'
RESULTS_ROOT = f'{ROOT}/results'

# ===================== FAULT INJECTION =====================
FAULT_MODEL = 'stuck-at_params'
SEED = 51195
MAX_FAULTS_TO_INJECT = 20000
BATCH_SIZE = 256
DEVICE = 'cuda'
DELETE_FAULTY_OUTPUT = True #if True it delete the faulty output folder after data analyzer to save space on the disk

# ===================== OTTIMIZZAZIONI RETE =====================
# Raramente usati — di solito il modello pruned/TMR viene caricato direttamente tramite NETWORK_NAME
PRUNING = False
TMR = False

# ===================== OPZIONI AVANZATE =====================
# Raramente usate
FORCE_RELOAD = False   # forza il ricalcolo del clean output
TRAIN = False         # usa il train set invece del test set
DRY_RUN = False       # esegui senza salvare output
PRINT = True          # stampa dettagli fault list

# ===================== OPZIONI RARE =====================
SEED_IMAGENET = 0.7      # usato solo per ImageNet, moltiplicato per 1000, 0.7 in data analyzer why
FORBID_CUDA = False    # forza l'uso della CPU
OUTPUTFMANALYZER = False  # analisi output feature maps
DEBUG = False          # stampa dettagli aggiuntivi