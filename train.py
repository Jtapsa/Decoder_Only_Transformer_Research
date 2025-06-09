import os
import time
import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Paformer_v3 import Decoder
from Paformer_v3 import RMS_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to all bucketed files
bucket_dir = "/projappl/project_2014349/Pre_Tokenized_Stream"


class ModelArgs_L:
    def __init__(self):
        self.vocab_size = 50278
        self.dim = 256
        self.n_layers = 2
        self.n_heads = 8
        self.n_groups = 2
        self.r = self.n_heads // self.n_groups
        self.ffn_dim = 4 * self.dim
        self.head_dim = self.dim // self.n_heads
        self.max_seq_len = 256
        self.batch_size = 32
        self.epochs = 2
        self.pad_id = 50277
        self.attention_type = "MultiH"
        self.ff_type = "SwiGLU"
        self.norm = "RMS"
        self.model_name = "Combined_Parallel_Gumb_MoE_wo_PF_TEST"
        self.device = device
        self.learning_rate = 5e-4
        self.weight_decay = 0.1
        self.betas = (0.9, 0.95)
        self.adam_eps = 1e-5
        
class ModelArgs_P:
    def __init__(self):
        self.dim = 128
        self.n_heads = 4
        self.ffn_dim = 4 * self.dim
        self.head_dim = self.dim // self.n_heads
        self.max_seq_len = 256
        self.attention_type = "MultiH"
        self.ff_type = "SwiGLU"
        self.norm = "RMS"
        self.n_parallel_blocks = 2
        self.device = device
        
        
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TokenDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.data.append(torch.LongTensor(obj["input_ids"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
class CombinedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, (nn.LayerNorm, RMS_norm)):
        with torch.no_grad():
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()

def load_emb_data(path_1, path_2):
    emb_1 = path_1["embeddings.weight"]
    emb_2 = path_2["embeddings.weight"]
    return torch.cat([emb_1, emb_2], dim=1)

def load_linear_data(path_1, path_2):
    lin_1 = path_1["Linear.linear.weight"]
    lin_2 = path_2["Linear.linear.weight"]
    return torch.cat([lin_1, lin_2], dim=1)
    
def load_pre_trained_weights(model):
    path_1 = torch.load("Path_1.pt", map_location=device)
    path_2 = torch.load("Path_2.pt", map_location=device)
    
    # Intializign Embedding weights to correspond Path_1 and Path_2 combination
    model.embeddings.weight.data = load_emb_data(path_1, path_2).clone()

    # Intializing "Path" weights
    for layer_id in [1, 2, 3]:
        parallel_layer = model.layers[layer_id]
    
        # Intializing parallel block[0] from Path_1
        for name, param in parallel_layer.blocks[0].named_parameters():
            full_name = f"layers.{layer_id-1}.{name}"
            if full_name in path_1:
                param.data.copy_(path_1[full_name])
    
        # Intializing parallel block[1] from Path_2
        for name, param in parallel_layer.blocks[1].named_parameters():
            full_name = f"layers.{layer_id-1}.{name}"
            if full_name in path_2:
                param.data.copy_(path_2[full_name])
                
    # Intializign last Linear weights to correspond Path_1 and Path_2 combination           
    model.Linear.linear.weight.data = load_linear_data(path_1, path_2).clone()


def evaluate_on_all_buckets(model, val_files, args):
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for path in val_files:
            dataset = TokenDataset(path)
            val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
            for batch in val_loader:
                batch = batch.to(args.device)
                labels = batch.clone()
                labels[:, :-1] = batch[:, 1:]
                logits, _, _ = model(batch)
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=args.pad_id)
                total_loss += loss.item()
                total_batches += 1
                
    model.train()
    avg_loss = total_loss / total_batches
    return avg_loss, math.exp(avg_loss)

def train(args,args_parallel, grad_accum_steps=8, use_scheduler=True, scheduler_type="cosine", checkpoint_epoch=None):
    set_random_seed()
    model = Decoder(args, args_parallel).to(args.device)
    model.apply(init_weights)
    load_pre_trained_weights(model)
    
    # Freeze pre-trained parameters form Path_1 and Path_2
    #for name, param in model.named_parameters():
        #if not any(x in name for x in ["Share", "MoE", "LargeLayer"]):
            #param.requires_grad = False
    
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} — {tuple(param.shape)}")

    val_files = sorted([os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) if f.startswith("validation_")])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay
    )

    loss_history = []
    val_loss_history = []
    val_ppl_history = []

    folder = args.model_name
    os.makedirs(folder, exist_ok=True)

    args_dict = vars(args)
    args_parallel_dict = vars(args_parallel)
    
    args_dict["device"] = str(args_dict["device"])
    args_parallel_dict["device"] = str(args_parallel_dict["device"])
    
    combined_args = {
        "ModelArgs": args_dict,
        "ModelArgs_Parallel": args_parallel_dict
    }

    
    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_files = sorted([os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) if f.startswith("train_bucket_comb") and f.endswith(f"epoch{epoch+1}.jsonl")])
        print(f"Found {len(train_files)} training buckets and {len(val_files)} validation buckets.")

        # Combine all data
        all_data = []
        for train_path in train_files:
            with open(train_path, "r") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    all_data.append(torch.LongTensor(obj["input_ids"]))
                    
        combined_dataset = CombinedDataset(all_data)
        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            combined_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            generator=g
        )
        if use_scheduler:
            if scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
            elif scheduler_type == "linear":
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / len(train_loader))
            else:
                scheduler = None
        else:
            scheduler = None

        total_loss = 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        milestones = {int(0.25 * num_batches), int(0.5 * num_batches), int(0.75 * num_batches)}

        for i, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            labels = batch.clone()
            labels[:, :-1] = batch[:, 1:]
                
            logits, entropy_total, load_total = model(batch)

            λ_entropy = 0.01
            λ_load = 0.01

            loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=args.pad_id)
            loss = loss +  λ_entropy * entropy_total + λ_load * load_total
            loss = loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item() * grad_accum_steps
            loss_history.append(loss.item() * grad_accum_steps)
            
            if i in milestones:
                pct = int(100 * i / num_batches)
                print(f"🟢 {pct}% of epoch completed...")
                
        avg_loss = total_loss / len(train_loader)
        print(f" Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")

        # Evaluate on validation buckets
        val_loss, val_ppl = evaluate_on_all_buckets(model, val_files, args)
        print(f"Validation | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        val_loss_history.append(val_loss)
        val_ppl_history.append(val_ppl)

        if checkpoint_epoch is not None and (epoch + 1) == checkpoint_epoch:
            ckpt = os.path.join(folder, f"{args.model_name}_checkpoint.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"[Checkpoint Saved] → {ckpt}")       
            with open(os.path.join(folder, f"{args.model_name}_loss_checkpoint.json"), "w") as f:
                json.dump(loss_history, f)
            with open(os.path.join(folder, f"{args.model_name}_val_metrics_checkpoint.json"), "w") as f:

                json.dump({"val_loss": val_loss_history, "val_perplexity": val_ppl_history}, f, indent=4)
            with open(os.path.join(folder, f"{args.model_name}_args_checkpoint.json"), "w") as f:
                json.dump(combined_args, f, indent=4)

    # Final save
    torch.save(model.state_dict(), os.path.join(folder, f"{args.model_name}.pt"))
    with open(os.path.join(folder, f"{args.model_name}_loss.json"), "w") as f:
        json.dump(loss_history, f)
    with open(os.path.join(folder, f"{args.model_name}_val_metrics.json"), "w") as f:
        json.dump({"val_loss": val_loss_history, "val_perplexity": val_ppl_history}, f, indent=4)
    with open(os.path.join(folder, f"{args.model_name}_args.json"), "w") as f:
        json.dump(combined_args, f, indent=4)


def main():
    args = ModelArgs_L()
    args_parallel = ModelArgs_P()
    
    print("ModelArgs Large Block Configuration:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")
        
    print("ModelArgs Parallel Block Configuration:")
    for key, value in args_parallel.__dict__.items():
        print(f"  {key}: {value}")
        
    try:
        train(args, args_parallel, checkpoint_epoch=1)
    except Exception as e:
        import traceback
        print("Training crashed:")
        traceback.print_exc()

if __name__ == "__main__":
    main()






