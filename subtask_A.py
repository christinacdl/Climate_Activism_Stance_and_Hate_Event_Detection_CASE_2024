# -*- coding: utf-8 -*-
import json
import os
import re
import torch
import wandb 
from copy import deepcopy
from huggingface_hub import notebook_login 
from datasets import load_dataset, concatenate_datasets 
from datasets import Dataset  
import warnings
from transformers.utils import logging
import evaluate
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import wordsegment as ws
import seaborn as sns
import emoji
import random
import re
import glob
from skimpy import skim
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, DataCollatorWithPadding
from huggingface_hub import login
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,  matthews_corrcoef
from typing import Callable
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
import torchvision
from sklearn.utils.class_weight import compute_class_weight
import datetime
import torch.nn as nn
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    LoraConfig,
    PeftModel,
    PeftConfig,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
    PromptEncoder,
    PrefixTuningConfig)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback, 
    pipeline,
    AutoConfig, 
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification, 
    AutoTokenizer)

# Ignore warnings
warnings.filterwarnings("ignore")
logging.set_verbosity(logging.CRITICAL) 

for i in range(torch.cuda.device_count()):
  info = torch.cuda.get_device_properties(i)
  print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Set up GPU for Training
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print(f'There are {torch.cuda.device_count()} GPU(s) available.')
  print('Device name:', torch.cuda.get_device_name(0))
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device('cpu')
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}") 

# os.environ["WANDB_PROJECT"] = "Subtask_A-LoRA" # log to your project
# os.environ["WANDB_LOG_MODEL"] = "all" # log your models

##==================== UTILS ====================##

# A function that sets seed for reproducibility
def set_seed(seed_value):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)


# A function that checks if a directory exists else creates the directory
def check_create_path(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print('Directory created at {}'.format(path))
  else:
    print('Directory {} already exists!'.format(path))


# A function that reads a csv or tsv file
def read_a_csv_tsv_file(filename):
  # Check first whether a certain file or directory exists
  if os.path.exists(filename):
    print('Current file opened: ',[os.path.join(filename, file) for file in glob.glob(filename)])

    # Find the file extension to open it properly
    find_separator = {'.csv': ',', '.tsv': '\t'}
    basename, format = os.path.splitext(filename)
    assert format in find_separator
    separator = find_separator[format]

    # Read different extensions of files using pandas with 2 different separators
    read_file = pd.read_csv(filename, sep = separator, encoding = 'utf-8')

    return read_file

  else:
    print('File or directory not accessible. Please check the filename and ensure that the entered path of the file is in "tsv" or "csv" form.')


def open_json_dataset(json_file, type_split = 1):
  # Open the json dataset
  if type_split == 1:
    dataset = load_dataset('json', data_files = json_file, split = 'train')
  elif type_split == 2:
    dataset = load_dataset('json', data_files = json_file, split = 'validation')
  elif type_split == 3:
    dataset = load_dataset('json', data_files = json_file, split = 'test')
  else:
    print('Please specify the number "1" for train set, "2" for validation and "3" to use the test set.')
  print(dataset)
  return dataset


# A function that opens and reads a dataset either from Hugging Face or from a local directory
def open_dataset(dataset_path, text_column, label_column, huggingface_dataset = True, json_dataset = False, csv_tsv_dataset = False, type_split = 1, clean_text = True, labelled_dataset = True):
  """Opens a dataframe object or Hugging Face dataset, or json dataset converts it into dataframe and presents an overview of values"""
  if huggingface_dataset:
    if type_split == 1:
      dataset = load_dataset(dataset_path, split='train')
      read_file = pd.DataFrame(dataset)
    elif type_split == 2:
      dataset = load_dataset(dataset_path, split='validation')
      read_file = pd.DataFrame(dataset)
    elif type_split == 3:
      dataset = load_dataset(dataset_path, split='test')
      read_file = pd.DataFrame(dataset)
  elif json_dataset:
    read_file = pd.read_json(dataset_path)
  elif csv_tsv_dataset:
    read_file = read_a_csv_tsv_file(dataset_path)
  else:
    print('Please specify whether it is a Hugging Face Dataset, a json dataset or a csv/tsv dataset. For the Hugging Face dataset, select type_split = "1" for train set, "2" for validation and "3" for the test set.')

  skimpy_file = skim(read_file)
  print(skimpy_file)

  if text_column != 'text':
    read_file = read_file.rename({text_column:'text'}, axis = 1)
  else:
    read_file
  
  if label_column is not None:
    if label_column != 'label':
      read_file = read_file.rename({label_column:'label'}, axis = 1)
    else:
      read_file
    print(read_file.label.value_counts())
  else:
    read_file

  print('Any missing values in the file:', read_file.isnull().values.any())
  print('Number of missing values in the file:', read_file.isnull().sum().sum())
  print('Number of duplicates in the file:', read_file.duplicated(subset = 'text').sum())
  
  if clean_text: 
    read_file.dropna(axis=1,how='all', inplace = True)
    print('Number of missing values in the file after cleaning:', read_file.isnull().sum().sum()) 
    read_file.drop_duplicates(subset = ['text'], keep = 'first', inplace = True) #, 
    print('Number of duplicates in the file after cleaning:', read_file.duplicated(subset = 'text').sum())
    read_file.reset_index(inplace = True) 
  else:
    read_file
  
  if label_column is not None:
    print(read_file.label.value_counts())
  else:
    read_file
  

  if labelled_dataset:
    # Encode the concatenated data
    encoded_texts = [tokenizer.encode(sent, add_special_tokens = True) for sent in read_file.text.values]
    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_texts])
    print('Initial maximum sentence length: ', max_len)
    # Find the minimum length
    min_len = min([len(sent) for sent in encoded_texts])
    print('Initial minimum sentence length: ', min_len)
  else:
    None
  return read_file


def compute_metrics(p):
  """Computes micro-F1 score, macro-F1 score, accuracy on a batch of predictions"""
  logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = np.argmax(logits, axis=1)
  macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
  micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
  accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
  return {'micro_f1': micro_f1, 'macro_f1': macro_f1, 'accuracy': accuracy}


# A function that calculates all the metrics using the validation/test set
def calculate_metrics(y_true, preds, class_names, save_directory_name):
  print('\nCALCULATING METRICS...')
  
  #assert len(preds) == len(y_true)
  # Calculate the accuracy of the model
  acc = accuracy_score(y_true, preds)
  # Calculate the Matthews Correlation Coefficient
  # -1 indicates total disagreement between predicted classes and actual classes
  # 0 is synonymous with completely random guessing, 1 indicates total agreement between predicted classes and actual classes
  mcc = matthews_corrcoef(y_true, preds)
  model_f1_score_micro = f1_score(y_true, preds, average = 'micro', zero_division = 1)
  model_precision_micro = precision_score(y_true, preds, average = 'micro', zero_division = 1)
  model_recall_micro = recall_score(y_true, preds, average = 'micro', zero_division = 1)
  model_f1_score_macro = f1_score(y_true, preds, average = 'macro', zero_division = 1)
  model_precision_macro = precision_score(y_true, preds, average = 'macro', zero_division = 1)
  model_recall_macro = recall_score(y_true, preds, average = 'macro', zero_division = 1)
  precision, recall, fscore, support = score(y_true, preds, zero_division = 1)
  print(f'Accuracy: {acc}')
  print(f'Micro-F1 Score: {model_f1_score_micro}')
  print(f'Macro-F1 Score: {model_f1_score_macro}') 
  print(f'Macro-Precision Score: {model_precision_macro}')
  print(f'Macro-Recall Score: {model_recall_macro}')
  print(f'Matthews Correlation Coefficient: {mcc}')
  print(f'\nPrecision of each class: {precision}')
  print(f'Recall of each class: {recall}')
  print(f'F1 score of each class: {fscore}')
  print(classification_report(y_true, preds, target_names = class_names, digits=4))
  # Create the confusion matrix
  cm = confusion_matrix(y_true, preds)
  df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels')
  plt.savefig(save_directory_name, bbox_inches='tight')
  #plt.show()
  plt.close()
  return model_f1_score_macro, model_f1_score_micro, fscore, acc, precision, recall, support

def tokenize(batch):
  return tokenizer(batch['text'], 
                   max_length = args['max_seq_length'], 
                   padding='max_length',
                   truncation=True,
                   add_special_tokens=True,
                   return_attention_mask=True,
                   return_tensors="pt") 

# PRE-PROCESSING
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize = ['user', 'url', 'email'],

    # terms that will be annotated
    #annotate = {'hashtag'},  #{'allcaps', 'repeated', 'elongated'},

    # corpus from which the word statistics are going to be used for word segmentation
    segmenter = 'twitter',  # or 'english'

    # corpus from which the word statistics are going to be used for spell correction
    corrector = 'twitter',  # or 'english'

    fix_html = False,              # fix HTML tokens
    fix_text = False,              # fix text
    unpack_hashtags = True,       # perform word segmentation on hashtags
    unpack_contractions = False,  # Unpack contractions (can't -> can not)
    spell_correct_elong = False,   # spell correction for elongated words

    tokenizer = SocialTokenizer(lowercase = False).tokenize)


ws.load()
def segment_hashtags(text):
  text = re.sub('#[\S]+', lambda match: ' '.join(ws.segment(match.group())), text)
  return text

def emojis_into_text(sentence):
  demojized_sent = emoji.demojize(sentence)
  emoji_txt = re.sub(r':[\S]+:', lambda x: x.group().replace('_', ' ').replace('-', ' ').replace(':', ''), demojized_sent)
  return emoji_txt

def preprocessing(text):
  # Convert the emojis into their textual representation
  text = emojis_into_text(text)

  # # Replace '&amp;' with 'and'
  text = re.sub(r'&amp;','and', text)
  text = re.sub(r'&','and', text)

  # # # Replace the unicode apostrophe
  text = re.sub(r"?","'", text)
  text = re.sub(r'?','"', text)
 
  # Replace consecutive non-ASCII characters with whitespace
  text = re.sub(r'[^\x00-\x7F]+',' ', text)

  text = re.sub(' +',' ', text) 

  # Apply the text processor from ekphrasis library
  text = ' '.join(text_processor.pre_process_doc(text))

  # Apply hashtag segmentation
  text = segment_hashtags(text)

  return text


# A function that splits the data into training and validation
def data_splitting(dataframe, text_column, label_column, split_ratio):
  x_train_texts, y_val_texts, x_train_labels, y_val_labels = train_test_split(dataframe[text_column], dataframe[label_column],
                                                                              random_state = 42,
                                                                              test_size = split_ratio,
                                                                              stratify = dataframe[label_column])
  print(f'Dataset split into train and validation/test sets using {split_ratio} split ratio.')
  train_df = pd.concat([x_train_texts, x_train_labels], axis = 1)
  val_df = pd.concat([y_val_texts, y_val_labels], axis = 1)
  print(f'Size of training set: {len(train_df)}')
  print(f'Size of validation/test set: {len(val_df)}')
  return train_df, val_df


def compute_class_weights(classes):
  weight1, weight2 = compute_class_weight(class_weight = 'balanced',
                                      classes = np.unique(classes),
                                      y = classes)
  print(f'Weight for class 0: {weight1}')
  print(f'Weight for class 1: {weight2}')
  return weight1, weight2


class CustomCallback(TrainerCallback):
  def __init__(self, trainer) -> None:
    super().__init__()
    self._trainer = trainer

  def on_epoch_end(self, args, state, control, **kwargs):
    if control.should_evaluate:
      control_copy = deepcopy(control)
      self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train") #####
      return control_copy

##==================== PARAMETERS & TOKENIZER INITIALIZATION ====================##
dict_text_classification_model_names = {1: 'bert-large-uncased',
                                        2: 'bert-base-multilingual-uncased',
                                        3: 'albert-xlarge-v1',
                                        4: 'albert-xlarge-v2',
                                        5: 'albert-xxlarge-v1',
                                        6: 'albert-xxlarge-v2',
                                        7: 'roberta-large',
                                        8: 'xlm-roberta-large',
                                        9: 'microsoft/deberta-large',
                                        10: 'microsoft/deberta-xlarge',
                                        11 : 'microsoft/deberta-v2-xlarge',
                                        12 : 'microsoft/deberta-v2-xxlarge',
                                        13 : 'microsoft/deberta-v3-large',
                                        14 : 'microsoft/mdeberta-v3-base',
                                        15 : 'GroNLP/hateBERT',
                                        16: 'vinai/bertweet-base',
                                        17: 'cardiffnlp/twitter-xlm-roberta-base'}

dict_causal_model_names = {1: 'bigscience/bloomz-560m',
                           2: 'NousResearch/Llama-2-7b-chat-hf',
                           3: 'NousResearch/Llama-2-13b-chat-hf',
                           4: 'microsoft/DialoGPT-medium',
                           5: 'microsoft/DialoGPT-large',
                           6: 'Open-Orca/Mistral-7B-OpenOrca', 
                           7: 'HuggingFaceH4/zephyr-7b-alpha', 
                           8: 'mistralai/Mistral-7B-v0.1', 
                           9: 'mistralai/Mistral-7B-Instruct-v0.1',
                           10: 'HuggingFaceH4/zephyr-7b-beta', 
                           11: 'NousResearch/Llama-2-7b-hf',
                           12: 'NousResearch/Llama-2-13b-hf',
                           13: 'TheBloke/Mistral-7B-OpenOrca-GPTQ',
                           14: 'TheBloke/Llama-2-7b-Chat-GPTQ',
                           15: 'TheBloke/zephyr-7B-beta-GPTQ',
                           16: 'TheBloke/Llama-2-7B-GPTQ',
                           17: 'Mistral-7B-v0.1',
                           18: 'decapoda-research/llama-7b-hf',
                           19: 'mistralai/Mixtral-8x7B-v0.1'}

dict_seq_seq_model_names = {1: 't5-large',
                            2: 'microsoft/GODEL-v1_1-large-seq2seq',
                            3: 'bigscience/mt0-large'}

dict_model_type = {1: 'TEXT_CLASSIF',
                   2: 'CAUSAL',
                   3: 'SEQ2SEQ',
                   4: 'TOKEN_CLASSIF',
                   5: 'IMAGE_CLASSIF'}

dict_task_type = {1: 'SEQ_CLS',
                  2: 'CAUSAL_LM',
                  3: 'SEQ_2_SEQ_LM',
                  4: 'TOKEN_CLS'}

dict_config_type = {1: 'PrefixTuning',
                    2: 'PromptTuning',
                    3: 'LoRA',
                    4: 'PromptEncoder'}

args = {'task_name': '=== Climate Activism Stance and Hate Event Detection Shared Task at CASE@EACL2024-SubTask A ===',
        'data_directory': '/home/ch.christodoulou/classification/Sub-task_A/',
        'results_data_directory': '/home/ch.christodoulou/classification/Sub-task_A/results/',
        'output_model_directory': '/home/ch.christodoulou/classification/Sub-task_A/outputs/',
        'model_name': str(dict_causal_model_names[8]), # Change the dictionary name and the index of the model of your choice
        'model_type': str(dict_model_type[1]), # Change the index for text classification/causal language modeling/Sequence2Sequence
        'task_type' : str(dict_task_type[1]), # Change the index for text classification/causal language modeling/Sequence2Sequence
        'new_model_name': 'Mistral-PromptTuning-Hate-Speech-Detection',
        'config': str(dict_config_type[2]), 
        'inference_mode': True,
        'num_virtual_tokens': 37, 
        'modules_to_save': ['classifier'], 
        'num_classes': 2,
        'max_seq_length': 195, 
        'data_split_ratio': 0.2,
        'train_batch_size': 16, 
        'validation_batch_size': 16,
        'num_train_epochs': 10, 
        'warmup_steps': 0,
        'weight_decay':  0.0001, 
        'learning_rate': 1e-4, 
        'adam_epsilon': 1e-8,
        'gradient_accumulation_steps': 2,
        'gradient_checkpointing': True,
        'max_grad_norm': 0.3, 
        'early_stopping_patience': 5,
        'seed': 42,
        'optimizer':'paged_adamw_32bit', 
        'lr_scheduler_type': 'constant', 
        'warmup_ratio': 0.1,
        'group_by_length': True,              # Group sequences into batches with same length. Saves memory and speeds up training considerably
        'save_steps': 1000,                   # Save checkpoint every X updates steps
        'logging_steps': 1000,
        'evaluation_strategy': 'epoch',
        'save_strategy':'epoch',
        'eval_steps': 1000,
        'save_total_limit': 2,
        'packing': False,                     # Pack multiple short examples in the same input sequence to increase efficiency
        'fp16': False,
        'bf16': False,
        'greater_is_better': True,
        'load_best_model_at_end': True,
        'overwrite_output_dir': True,
        'push_to_hub': True,
        'report_to': 'wandb', 
        'hub_strategy': 'every_save',
        'ignore_pad_token_for_loss': True,
        'problem_type': 'single_label_classification',
        'prompt_tuning_init_text': "[INST]Your task is to classify if the text contains hate speech or not, and return the answer as the corresponding label '0' or '1'[/INST]",

        # bitsandbytes parameters
        'use_4bit': True,                     # Activate 4-bit precision base model loading
        'bnb_4bit_compute_dtype': 'float16',  # Compute dtype for 4-bit base models
        'bnb_4bit_quant_type': 'nf4',         # Quantization type (fp4 or nf4)
        'use_nested_quant': False,            # Activate nested quantization for 4-bit base models (double quantization)

        # QLora Parameters
        'lora_r': 16,                          # LoRA attention dimension
        'lora_alpha': 16,                     # Alpha parameter for LoRA scaling
        'lora_dropout': 0.05,                 # Dropout probability for LoRA layers 
        }

print('================',str(args['task_name']),'================\n')

# Get the directory names and the specific model used
print('Output directory: ' + str(args['output_model_directory']))
print('Model Name: ' + str(args['model_name']))
args['output_specific_model_dir'] = args['output_model_directory'] + args['model_name'] + '/' 
print('Output Directory: ' + str(args['output_specific_model_dir']))

# Check whether the directories exist else create them
print('\nChecking that the necessary paths exist...')
check_create_path(args['data_directory'])
check_create_path(args['output_model_directory'])
check_create_path(args['results_data_directory'])
check_create_path(args['output_specific_model_dir'])

repository_id = args['new_model_name'] 

MODEL_CLASSES = {'TEXT_CLASSIF': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
                 'CAUSAL' : (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
                 'SEQ2SEQ': (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer),
                 'TOKEN_CLASSIF' : (AutoConfig, AutoModelForTokenClassification, AutoTokenizer),
                 'IMAGE_CLASSIF' : (AutoConfig, AutoModelForImageClassification, AutoTokenizer)}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, args['bnb_4bit_compute_dtype'])

bnb_config = BitsAndBytesConfig(
    load_in_4bit = args['use_4bit'],
    bnb_4bit_quant_type = args['bnb_4bit_quant_type'],
    bnb_4bit_compute_dtype = args['bnb_4bit_compute_dtype'],
    bnb_4bit_use_double_quant = args['use_nested_quant'])

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and args['use_4bit']:
  major, _ = torch.cuda.get_device_capability()
  if major >= 8:
    print("=" * 80)
    print("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)

# Set seed for reproducibility
set_seed(args['seed'])

tokenizer = tokenizer_class.from_pretrained(args['model_name'], add_prefix_space=False,
                                            use_fast = True, trust_remote_code=True, add_eos_token=True, add_bos_token = True, padding_side = 'left',
                                            do_lower_case = False)  

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(['<user>', '<url>', '<email>'], special_tokens = True)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest')

tokens = tokenizer.tokenize(args['prompt_tuning_init_text'])
num_tokens = len(tokens)
print(f"The prompt contains {num_tokens} tokens.")

##==================== LABELS OF SUB-TASKS ====================##
label2id = {'NON-HATE': 0, 'HATE': 1}
id2label = {0: 'NON-HATE', 1: 'HATE'}
class_names = ['NON-HATE', 'HATE']

##==================== OPEN AND PROCESS DATASETS ====================##
training_data = open_dataset(args['data_directory'] + 'SubTask-A-train.csv', 
                            'tweet', 'label', huggingface_dataset = False, json_dataset = False, csv_tsv_dataset = True, 
                             type_split = 1, clean_text= True, labelled_dataset = True)

training_data['text'] = training_data['text'].apply(lambda x: preprocessing(x))

# Encode the concatenated data
encoded_texts = [tokenizer.encode(sent, add_special_tokens = True) for sent in training_data['text'].values]
# Find the maximum length
max_len = max([len(sent) for sent in encoded_texts])
print('Maximum sentence length: ', max_len)

print(training_data['text'][0])
print(training_data['text'][8])

counts = training_data['label'].value_counts()
minority_class = counts.idxmin()
print(f'The minority class is: {minority_class}') 

# df1 = pd.read_csv(args['data_directory'] + 'SubTask-A-(index,tweet)val.csv')
# df2 = pd.read_csv(args['data_directory'] + 'SubTask-A(index,label)val.csv')
# data = pd.merge(df1, df2, on='index')
# # print(data)
# data.to_csv(args['data_directory'] + 'SubTask-A_labelled_val.csv', index=False)

val_data = open_dataset(args['data_directory'] + 'SubTask-A_labelled_val.csv', 
                                  'tweet', 'label', huggingface_dataset = False, json_dataset = False, csv_tsv_dataset = True, 
                                   type_split = 1, clean_text= True, labelled_dataset = True)

val_data['text'] = val_data['text'].apply(lambda x: preprocessing(x))

# df1 = pd.read_csv(args['data_directory'] + 'SubTask-A-(index,tweet)test.csv')
# df2 = pd.read_csv(args['data_directory'] + 'SubTask-A(index,label)test.csv')
# data = pd.merge(df1, df2, on='index') 
# data.to_csv(args['data_directory'] + 'SubTask-A_labelled_test.csv', index=False)

test_data = open_dataset(args['data_directory'] + 'SubTask-A-(index,tweet)test.csv', 
                                  'tweet', None, huggingface_dataset = False, json_dataset = False, csv_tsv_dataset = True, 
                                   type_split = 1, clean_text= False, labelled_dataset = True)

test_data['text'] = test_data['text'].apply(lambda x: preprocessing(x))

# Encode the concatenated data
encoded_texts_test = [tokenizer.encode(sent, add_special_tokens = True) for sent in test_data['text'].values]
# Find the maximum length
max_len_test = max([len(sent) for sent in encoded_texts_test])
print('Maximum sentence length: ', max_len_test)

train_dataset = Dataset.from_pandas(training_data, split='train')
encoded_train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    num_proc=1,
    remove_columns = ['text', 'index','level_0'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset")
encoded_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
print(f"Keys of tokenized dataset: {list(encoded_train_dataset.features)}")

validation_dataset = Dataset.from_pandas(val_data, split='validation')
encoded_val_dataset = validation_dataset.map(tokenize,
    batched=True,
    num_proc=1,
    remove_columns = ['text', 'index'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset")
encoded_val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) 

test_dataset = Dataset.from_pandas(test_data, split='test')
encoded_test_dataset = test_dataset.map(
    tokenize,
    batched=True,
    num_proc=1,
    remove_columns = ['text', 'index'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset")   
encoded_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask']) 

tokenized_inputs = concatenate_datasets([train_dataset, validation_dataset, test_dataset]).map(
    lambda x: tokenizer(x["text"], truncation=True), batched=True, remove_columns=['text', 'label', 'index'])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# Calculate the total number of samples
class_counts = training_data['label'].value_counts()
total_samples = sum(class_counts.values)
# Calculate the weights
weights = {class_id: total_samples / num_samples_in_class for class_id, num_samples_in_class in class_counts.items()}
print(weights) 


def get_weighted_trainer(classes):
  # You can use this weights if you want to balance the classes
  #weights_class_1, weight_class_2 = compute_class_weights(classes)
    
  class _WeightedBCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
      labels = inputs.pop("labels")     
      outputs = model(**inputs)
      logits = outputs.get("logits")
      loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([weights[0], weights[1]], device=labels.device, dtype=logits.dtype)) 
      loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
      return (loss, outputs) if return_outputs else loss
  return _WeightedBCELossTrainer

##==================== MODEL INITIALIZATION ====================##
model = model_class.from_pretrained(args['model_name'], 
                                    num_labels = args['num_classes'], 
                                    id2label = id2label,
                                    label2id = label2id,
                                    device_map = 'auto', 
                                    offload_folder = 'offload',
                                    trust_remote_code = True, 
                                    torch_dtype = torch.float16,
                                    quantization_config = bnb_config)

model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = model.config.eos_token_id

if args['config'] == 'PrefixTuning':
  '''
  Prefix tuning is an additive method where only a sequence of continuous task-specific vectors is attached to the beginning of the input, or prefix. 
  Only the prefix parameters are optimized and added to the hidden states in every layer of the model. The tokens of the input sequence can still attend to the prefix as virtual tokens. 
  As a result, prefix tuning stores 1000x fewer parameters than a fully finetuned model, which means you can use one LLM for many tasks.
  '''
  peft_config = PrefixTuningConfig(task_type = args['task_type'], inference_mode = args['inference_mode'], num_virtual_tokens = args['num_virtual_tokens'])

elif args['config'] == 'PromptTuning':
  '''
  Prompting helps guide language model behavior by adding some input text specific to a task. Prompt tuning is an additive method for only training and updating the newly added prompt tokens to a pretrained model. 
  This way, you can use one pretrained model whose weights are frozen, and train and update a smaller set of prompt parameters for each downstream task instead of fully finetuning a separate model. 
  As models grow larger and larger, prompt tuning can be more efficient, and results are even better as model parameters scale.
  '''
  peft_config = PromptTuningConfig(task_type = args['task_type'],
                                    prompt_tuning_init = PromptTuningInit.TEXT,
                                    num_virtual_tokens = args['num_virtual_tokens'],
                                    prompt_tuning_init_text= args['prompt_tuning_init_text'],
                                    tokenizer_name_or_path = args['model_name'])
elif args['config'] == 'LoRA':
  '''
  LoRA, a technique that accelerates the fine-tuning of large models while consuming less memory. To make fine-tuning more efficient, LoRA's approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. 
  These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn't receive any further adjustments. 
  To produce the final results, both the original and the adapted weights are combined.
  '''
  peft_config = LoraConfig(lora_alpha = args['lora_alpha'], 
                           lora_dropout = args['lora_dropout'], 
                           r = args['lora_r'], 
                           bias = 'none',
                           target_modules = ['q_proj','v_proj'],
                           task_type = args['task_type'])

# P-tuning is a method for automatically searching and optimizing for better prompts in a continuous space 
elif args['config'] == 'PromptEncoder':
  peft_config = PromptEncoderConfig(task_type = args['task_type'], num_virtual_tokens = args['num_virtual_tokens'], encoder_hidden_size=128)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

##==================== RUN EXPERIMENTS ====================##
notebook_login()

# Set training parameters
arguments = TrainingArguments(
    output_dir = repository_id,
    logging_dir = f'{repository_id}/logs',
    evaluation_strategy = args['evaluation_strategy'],
    save_strategy = args['save_strategy'],
    eval_steps = args['eval_steps'],
    save_total_limit = args['save_total_limit'],
    learning_rate = args['learning_rate'],
    num_train_epochs = args['num_train_epochs'],
    metric_for_best_model = 'macro_f1',
    greater_is_better = args['greater_is_better'],
    weight_decay = args['weight_decay'],
    load_best_model_at_end = args['load_best_model_at_end'],
    per_device_train_batch_size = args['train_batch_size'],
    per_device_eval_batch_size = args['validation_batch_size'],
    overwrite_output_dir = args['overwrite_output_dir'],
    fp16 = args['fp16'],
    bf16 = args['bf16'],
    seed = args['seed'],
    warmup_ratio = args['warmup_steps'],
    gradient_accumulation_steps = args['gradient_accumulation_steps'],
    gradient_checkpointing = args['gradient_checkpointing'],
    optim = args['optimizer'],
    save_steps = args['save_steps'],
    logging_strategy = 'steps',
    logging_steps = args['logging_steps'],
    max_grad_norm = args['max_grad_norm'],
    group_by_length = args['group_by_length'],
    lr_scheduler_type = args['lr_scheduler_type'],
    report_to = args['report_to'],
    push_to_hub = args['push_to_hub'],
    hub_strategy = args['hub_strategy'])

weighted_trainer = get_weighted_trainer(training_data['label'])

trainer = weighted_trainer(  
     model = model,
     data_collator = data_collator,
     tokenizer = tokenizer,
     args = arguments,
     train_dataset = encoded_train_dataset,        
     eval_dataset = encoded_val_dataset,        
     compute_metrics = compute_metrics,
     callbacks = [EarlyStoppingCallback(early_stopping_patience = args['early_stopping_patience'])])

##==================== TRAIN & EVALUATE ====================##
print('TRAINING...')
trainer.add_callback(CustomCallback(trainer)) 
train_result = trainer.train()

tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
trainer.push_to_hub() 

# Train metrics
metrics = train_result.metrics
metrics['train_samples'] = len(encoded_train_dataset)
trainer.save_model()
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()

print('EVALUATING...')
# Evaluate on labelled data
metrics = trainer.evaluate(eval_dataset = encoded_val_dataset)
max_eval_samples = len(encoded_val_dataset)
metrics['eval_samples'] = max_eval_samples
trainer.log_metrics('eval', metrics)
trainer.save_metrics('eval', metrics)

##==================== GET PREDICTIONS & METRICS ====================##
print('PREDICTING LABELLED VALIDATION DATA...')
predictions, labels, metrics = trainer.predict(encoded_val_dataset, metric_key_prefix='predict')
max_predict_samples = len(encoded_val_dataset)
metrics['predict_samples'] = len(encoded_val_dataset)
trainer.log_metrics('predict', metrics)
trainer.save_metrics('predict', metrics)

preds = np.argmax(predictions, axis=-1)

# Calculate performance metrics on test set
macro_f1, micro_f1, f1_class, accuracy, precision_class, recall_class, support = calculate_metrics(labels, preds, class_names, f'{repository_id}/en_test_set_matrix.png')

df_true = pd.DataFrame(labels, columns = ['True'])
df_preds = pd.DataFrame(preds, columns = ['Prediction'])
df_metrics = pd.DataFrame([[macro_f1, micro_f1, accuracy, f1_class, precision_class, recall_class, support]],
                            columns = ['Macro_F1', 'Micro_F1', 'F1s', 'Accuracy', 'Precision', 'Recall', 'Support'])

# Concatenate id, text, true labels and predicted labels
final_true_preds = pd.concat([df_true, df_preds], axis = 1)

final_true_preds.to_csv(f'{repository_id}/subtaskA_hate_True_Predictions.csv', encoding = 'utf-8', index = False, header = True, sep =',')
df_metrics.to_csv(f'{repository_id}/subtaskA_hate_Results_Metrics.csv', encoding = 'utf-8', index = False, header = True, sep =',')

# print('PREDICTING UNLABELLED VALIDATION DATA...')
# unlabelled_val_predictions = trainer.predict(encoded_val_un_dataset)

# # Access the predictions
# unlabelled_val_predictions = unlabelled_val_predictions.predictions
# unlabelled_val_predictions_preds = np.argmax(unlabelled_val_predictions, axis=-1)
# unlabelled_val_predictions_preds = unlabelled_val_predictions_preds.flatten()
# df_unlabelled_val_predictions = pd.DataFrame(unlabelled_val_predictions_preds, columns = ['Label_pred'])

# # Create a list of dictionaries for the submission
# submission_dict = [{"index": idx, "prediction": pred} for idx, pred in zip(unlabelled_val_data['index'], df_unlabelled_val_predictions['Label_pred'])]
# # Sort the list of dictionaries by 'index' in ascending order
# submission_dict_sorted = sorted(submission_dict, key=lambda x: x['index'])
# # Write to a JSON file
# with open(f'{repository_id}/submission_subtaskA_val.json', 'w') as f:
#   for item in submission_dict_sorted: 
#     f.write(json.dumps(item) + '\n')

print('PREDICTING UNLABELLED TEST DATA...')
test_predictions = trainer.predict(encoded_test_dataset)

# Access the predictions
test_predictions = test_predictions.predictions
test_preds = np.argmax(test_predictions, axis=-1)
test_preds = test_preds.flatten()
df_test_predictions = pd.DataFrame(test_preds, columns = ['Label_pred'])

# Create a list of dictionaries for the submission
submission_dict_test = [{"index": idx, "prediction": pred} for idx, pred in zip(test_data['index'], df_test_predictions['Label_pred'])]
# Sort the list of dictionaries by 'index' in ascending order
submission_dict_sorted_test = sorted(submission_dict_test, key=lambda x: x['index'])
# Write to a JSON file
with open(f'{repository_id}/submission_subtaskA_test.json', 'w') as f:
  for item in submission_dict_sorted_test: 
    f.write(json.dumps(item) + '\n') 
