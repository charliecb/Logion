# Adapted from https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import getpass
import sys
epochs = 100

### ADJUST PARAMETERS AS DESIRED ###
num_logs_per_epoch = 2
num_evals_per_epoch = 2
num_saves_per_epoch = 1.5
mask_proportion = 0.15

main_directory = 'Greek_PH'    # This should be located inside the scratch work drive, /scratch/gpfs/<username>/
#data_subdirectory = 'data'

if len(sys.argv) == 1:
        print("Received no argument for batch size. Defaulting to 16.")
        batch_size = 16
elif len(sys.argv) > 1:
        print(f"Setting batch size to {sys.argv[1]}.")
        batch_size = int(sys.argv[1])
### ADJUST BASED ON INPUT DATA SIZE ###
num_steps = 450 # steps per epoch, currently hardcoded
log_every = int(num_steps/num_logs_per_epoch)
eval_every = int(num_steps/num_evals_per_epoch)
save_every = int(num_steps/num_saves_per_epoch)

username = getpass.getuser()
filestem = '/scratch/gpfs/' + username + '/' + main_directory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

# Assuming you want to use the pre-trained weights from this model, use given preload_path
# If you would like to train from scratch, specify a BertConfig to define a model
preload_path = 'cabrooks/LOGION-50k_wordpiece'
tokenizer = BertTokenizer.from_pretrained(preload_path)
model = BertForMaskedLM.from_pretrained(preload_path).to(device)

# In the current directory, we expect two files, train_example.txt and val_example.txt
# Each of these files are should contain newline-separated training examples (chunks of text containing <= 512 tokens)

with open('./data/train_example.txt', 'r') as f:
  text = f.read().split('\n')
  text.pop(-1)
  f.close()

train_inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
train_inputs['labels'] = train_inputs.input_ids.detach().clone()

with open('./data/val_example.txt', 'r') as f:
  text_val = f.read().split('\n')
  text_val.pop(-1)
  f.close()

val_inputs = tokenizer(text_val, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
val_inputs['labels'] = val_inputs.input_ids.detach().clone()

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = BaseDataset(train_inputs)
val_dataset = BaseDataset(val_inputs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_proportion)

training_args = TrainingArguments(
    evaluation_strategy = "steps",
    eval_steps=eval_every,
    logging_steps=log_every,
    save_steps=save_every,
    output_dir=filestem + '/example_run' + str(batch_size),
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(filestem + '/content_example_run' + str(batch_size) + '/tester')
