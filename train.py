from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import getpass
import sys

epochs = 80
#training_portion = 0.9      # Reserve 10% of dataset for testing
#random_seed = 42            # This seed is used so that we can recover the exact train/test split after this script terminates.
num_logs = 100            # (in steps)
num_evals = 100          # (in steps)
num_saves = 10
mask_proportion = 0.15

main_directory = 'Greek'    # This should be located inside the scratch work drive, /scratch/gpfs/<username>/
#data_subdirectory = 'data'

if len(sys.argv) == 1:
        print("Received no argument for batch size. Defaulting to 16.")
        batch_size = 16
elif len(sys.argv) > 1:
        print(f"Setting batch size to {sys.argv[1]}.")
        batch_size = int(sys.argv[1])

num_steps = int(35396/batch_size)
log_every = int(num_steps/num_logs)
eval_every = int(num_steps/num_evals)
save_every = int(num_steps/num_saves)

username = getpass.getuser()
filestem = '/scratch/gpfs/' + username + '/' + main_directory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

tokenizer = BertTokenizer.from_pretrained('pranaydeeps/Ancient-Greek-BERT')
model = BertForMaskedLM.from_pretrained('pranaydeeps/Ancient-Greek-BERT').to(device)

with open('ps_train.txt', 'r') as f:
  text = f.read().split('\n')
  text.pop(-1)
  f.close()

train_inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
train_inputs['labels'] = train_inputs.input_ids.detach().clone()

with open('ps_val.txt', 'r') as f:
  text_val = f.read().split('\n')
  text_val.pop(-1)
  f.close()

val_inputs = tokenizer(text_val, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
val_inputs['labels'] = train_inputs.input_ids.detach().clone()

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
    output_dir=filestem + '/out' + str(batch_size),
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

model.save_pretrained(filestem + '/content' + str(batch_size) + '/tester')