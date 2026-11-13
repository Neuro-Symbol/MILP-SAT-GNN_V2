
from collections import defaultdict
import gc
import humanize as H
import json
import numpy as np
import os
from os.path import join
import psutil
import resource
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import psutil
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchinfo import summary
from tqdm import tqdm
import yaml
import torch.nn as nn
import sys
import learners.common as LC
import math
import wandb
import gc

from learners.focal_loss import FocalLoss

from data.data_utils import (
    load_dataset_splits, load_dataset_splits_g4s
  )  # , GnnCustomDataset2, compute_metrics

from utils.misc import (
    set_device
)

from utils.math_utils import (
    scatter_add, 
    scatter_mul
    )

epsilon = 1e-7

class UnsupLoss2(nn.Module):
    def forward(self, out):
        loss = torch.sum(out)
        return loss
    
def eval_formula_scatter(values, batch, device):
    values = values.to(device).view(-1)

    edge_index = batch["left", "to", "right"].edge_index
    edge_attr = batch["left", "to", "right"].edge_attr
    clause_indices = edge_index[0]
    var_indices = edge_index[1]
    num_clauses = batch["left"].x.size(0)
    num_formulas = batch.num_graphs

    var_probs = values[var_indices]
    signs = edge_attr.squeeze()
    lit_values = torch.where(signs > 0, var_probs, 1.0 - var_probs)
    clause_values = 1.0 - scatter_mul(1.0 - lit_values, clause_indices, num_clauses)
    safe_clause_values = torch.clamp(clause_values, min=epsilon, max=1.0)
    return -scatter_add(torch.log(safe_clause_values), batch["left"].batch, num_formulas)


def nodeLevelPrediction(values, batch, device):
    return eval_formula_scatter(values, batch, device)


def sat_predictions(values, batch, device):
    binary_values = (values > 0.5).float()
    return (nodeLevelPrediction(binary_values, batch, device) == 0).float()


# target smoothing
def process_target(target):
    return 0.9 * target + 0.05

def train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        scheduler=None,
        epochs=100, 
        best_model_path="best_model.pth", 
        final_model_path="final_model.pth", 
        device=torch.device("cpu"), 
        train_freq=5,
        val_freq=1, 
        model_selection_mode="standard",
        model_selection_measure="accuracy",
        dev=False,
        model_name="GCNPolicy2",
        loss_function_name="mse",
        msgs=[],
        output_dir=None,
        classification_method="graph",
        formula_eval_method="unsup2",
        model_selection_threshold=0.05,
        clip_grad_norm=None
    ):

    print(device)
    model.to(device)

    
    # Use gradient clipping
    max_grad_norm = 1.0
    
    #prepare folders for saving models
    
    train_results = defaultdict(list)
    train_results["has_validation"] = len(val_loader) > 0

    best_epoch = -1
    best_val_acc = -float('inf')
    best_val_loss_at_best_acc = float('inf')
    
    initial_val = float('inf') if model_selection_measure == "loss" else -float('inf')
    best_val_measure = initial_val
    best_val_measure_within_gap = initial_val
    best_adj_score = initial_val

    train_cm = None
    val_cm = None
    
    classification_method = model.classification_method

    if formula_eval_method != "unsup2":
        raise ValueError(f"Unknown formula evaluation method: {formula_eval_method}")
    if classification_method == "node" and loss_function_name != "unsup2":
        print(f"Node-level unsup2 training overrides loss_function_name from '{loss_function_name}'")
        loss_function_name = "unsup2"

    match   loss_function_name: 
        case "mse":
            loss_fn = torch.nn.MSELoss()
        case "bce":
            loss_fn = torch.nn.BCEWithLogitsLoss()
        case "focal":
            loss_fn = FocalLoss()
            loss_fn = loss_fn.to(device)
        case "unsup2":
            loss_fn = UnsupLoss2()
        case _:
            assert False, f"Unknown loss function: {loss_function_name}"
    


    num_batches = len(train_loader)
    train_loss = float('inf')
    train_losses = []
    val_loss = float('inf')
    train_metrics = None

    # Memory occupied strictly by active tensors (in bytes), excluding allocator cache
    if torch.device(device).type == "mps":
        allocated_mb = torch.mps.current_allocated_memory() / 1e6
        print(f"Allocated MPS memory: {allocated_mb:.2f} MB")
        # Total memory allocated by the Metal driver, including caching pools
        driver_mb = torch.mps.driver_allocated_memory() / 1e6
        print(f"MPS Driver Total: {driver_mb:.2f} MB")


    for epoch in range(epochs):

        # Empty the MPS Cache:

        print("-----------------------------------------")
        print(f"Epoch {epoch+1}/{epochs} - START")
        print("Training set:", len(train_loader.dataset))
        print("Validation set:", len(val_loader.dataset))
        print("Batch size:", train_loader.batch_sampler.batch_size)
        print("Number of batches:", num_batches)
        print("Using loss:", loss_function_name)
        print("Using aggregation:", model.aggregation)
        print("Using scheduler:", scheduler is not None)
        print("Scheduler type:", scheduler.__class__.__name__)
        print("Learning rate: ", optimizer.param_groups[0]["lr"])

        # Memory occupied strictly by active tensors (in bytes), excluding allocator cache
        if torch.device(device).type == "mps":
            allocated_mb = torch.mps.current_allocated_memory() / 1e6
            print(f"Allocated MPS memory: {allocated_mb:.2f} MB")
            # Total memory allocated by the Metal driver, including caching pools
            driver_mb = torch.mps.driver_allocated_memory() / 1e6
            print(f"MPS Driver Total: {driver_mb:.2f} MB")
        # Memory usage
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"[RAM] Peak RSS: {max_rss_mb:.2f} MB")
        print("-----------------------------------------")
        # --
        
        for m in msgs:
            print(m)
        print()
        model.train()
        pbar = tqdm(train_loader, total=num_batches, position=0)
        

        running_loss = 0.0
        total_examples = 0
    

        for batch_idx, batch in enumerate(pbar):



            # print first example in batch

            if model_selection_mode == "standard":
                measure_str = f"val_{model_selection_measure} {best_val_measure:.3f}"
            elif model_selection_mode == "AdjScore":
                measure_str = f"AdjScore {best_adj_score:.3f}"
            elif model_selection_mode == "GenGap":
                best_val = best_val_measure_within_gap if best_val_measure_within_gap != initial_val else best_val_measure
                measure_str = f"val_{model_selection_measure} {best_val:.3f}"
            else:
                measure_str = f"val_acc {best_val_acc:.3f}"

            pbar.set_description(f"E {epoch+1:02d}/{epochs} - Best at {best_epoch:02d} with {measure_str}. Train loss: {train_loss:.3f}. Validation loss: {val_loss:.3f}")
            optimizer.zero_grad()

            # print shapes
            if False:
                print(f"x.shape: {x.shape}")
                print(f"edge_index.shape: {edge_index.shape}")
                print(f"edge_attr.shape: {edge_attr.shape}")
                print(f"batch.shape: {batch.shape}")
                print(f"labels.shape: {labels.shape}")

            # Forward pass
            batch = batch.to(device)
            out = model(batch)

            labels = batch.label  # .to(device)


            
            match classification_method:    
                case "graph": # graph level classification
                    out = out.squeeze(-1)
                case "node": # node level classification
                    out = out.squeeze(-1)
                    out = torch.sigmoid(out) # normalise in [0,1]
                    out = nodeLevelPrediction(out, batch, device)
                case _:        
                    print("learning.py: train_model: default classification case undefined")    
                    sys.exit(0)




            match loss_function_name: 
                case "mse":
                    loss = loss_fn(torch.sigmoid(out), labels)
                case "bce":
                    loss = loss_fn(out, labels)
                case "focal":
                    loss = loss_fn(out, labels)
                case "unsup2":
                    loss = loss_fn(out)
                case _:
                    assert False, f"Unknown loss function: {loss_function_name}"    


                
            # Backpropagation with gradient clipping

            loss.backward()

            # Backpropagation with GRADIENT CLIPPING, L2 norm
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            
            # --- PRINT GRADIENTS ---
            



            optimizer.step()

            running_loss += ( loss.item() * len(batch) )
            total_examples += len(batch)
            

            # del batch, out, loss
            
        train_loss = running_loss / total_examples


        train_losses.append(train_loss)

        if wandb.run:
            wandb.log({"train/loss": train_loss}, step=epoch)
        

        if (epoch+1) % train_freq == 0:
            train_results["train_validation_epochs"].append(epoch)
            print(f"-------> Epoch {epoch+1}/{epochs} - TRAIN Prediction")
            train_preds, _ = make_predictions(model, train_loader, device, loss_fn=None)


            train_metrics, train_cm = compute_metrics(pd.DataFrame.from_dict(train_preds))
            

            for k, v in train_metrics.items():
                tqdm.write(f"(prediction on training set) \t-  {k}: {v}")
                train_results[f"train/{k}"].append( v )
                if wandb.run:
                    wandb.log({f"train/{k}": v}, step=epoch)
        else:
            train_acc = None
            train_metrics = None
        
        save_model = False
        if (len(val_loader) == 0) and (train_loss < best_val_loss):
            print("-------> No validation set - choosing best model with train loss")
            best_val_loss = train_loss
            best_epoch = epoch+1
            save_model = True
            assert False, "Never here"

        if (len(val_loader)>0) and ( (epoch+1) %  val_freq == 0):
            train_results["validation_epochs"].append(epoch)

            print(f"-------> Epoch {epoch+1}/{epochs} - VALIDATION Prediction")
            res_dict, losses = make_predictions(model, val_loader, device, loss_fn, loss_function_name)
            res_df = pd.DataFrame.from_dict(res_dict)
            

            val_loss = np.mean(losses)
            train_results["validation_loss"].append( val_loss )
            val_metrics, val_cm = compute_metrics(res_df)
            if wandb.run:
                log_dict = {"val/loss": val_loss}
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log(log_dict, step=epoch)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if scheduler.mode == "min":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(val_metrics["accuracy"])
                else:
                    scheduler.step()
                
            print(pd.DataFrame(val_cm, columns=["Pred 0", "Pred 1"], index=["0", "1"]).to_markdown())
            tqdm.write("* VALIDATION METRICS")
            for k, v in val_metrics.items():
                tqdm.write(f"\t-  {k}: {v}")
                train_results[k].append( v )
            
            val_acc = val_metrics["accuracy"]

            if model_selection_measure == "loss":
                val_measure = val_loss
                train_measure = train_loss
                is_better = val_measure < best_val_measure
                is_better_within_gap = val_measure < best_val_measure_within_gap
            else:
                val_measure = val_metrics[model_selection_measure]
                if train_metrics is not None and model_selection_measure in train_metrics:
                    train_measure = train_metrics[model_selection_measure]
                else:
                    train_measure = train_loss if model_selection_measure == "loss" else val_measure
                is_better = val_measure > best_val_measure
                is_better_within_gap = val_measure > best_val_measure_within_gap

            if model_selection_mode == "standard":
                if is_better:
                    best_val_measure = val_measure
                    best_epoch = epoch+1
                    save_model = True
            elif model_selection_mode == "accuracy":
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch+1
                    save_model = True
            elif model_selection_mode == "accuracy_loss":
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss_at_best_acc = val_loss
                    best_epoch = epoch+1
                    save_model = True
                elif val_acc == best_val_acc:
                    if val_loss < best_val_loss_at_best_acc:
                        best_val_loss_at_best_acc = val_loss
                        best_epoch = epoch+1
                        save_model = True
            elif model_selection_mode == "GenGap":
                gap = abs(train_measure - val_measure)
                if gap <= model_selection_threshold:
                    if is_better_within_gap:
                        best_val_measure_within_gap = val_measure
                        best_epoch = epoch+1
                        save_model = True
                else:
                    if best_val_measure_within_gap == initial_val and is_better:
                        best_val_measure = val_measure
                        best_epoch = epoch+1
                        save_model = True
            elif model_selection_mode == "AdjScore":
                gap = abs(train_measure - val_measure)
                if model_selection_measure == "loss":
                    adj_score = val_measure + (model_selection_threshold * gap)
                    if adj_score < best_adj_score:
                        best_adj_score = adj_score
                        best_epoch = epoch+1
                        save_model = True
                else:
                    adj_score = val_measure - (model_selection_threshold * gap)
                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_epoch = epoch+1
                        save_model = True
                        
        if save_model:
            if 'val_acc' in locals():
                best_val_acc = val_acc
            print(f"New best at epoch {best_epoch} - mode: {model_selection_mode}")
            torch.save(model.state_dict(), best_model_path)
        
            if output_dir is not None:
                if train_cm is not None:
                    df_train_cm = pd.DataFrame(train_cm, columns=["Pred 0", "Pred 1"], index=["0", "1"])
                    df_train_cm.to_csv(os.path.join(output_dir, "best_train_confusion_matrix.csv"), index=True)
                    with open(os.path.join(output_dir, "best_train_confusion_matrix.md"), "w") as f:
                        f.write(df_train_cm.to_markdown())
                if val_cm is not None:
                    df_val_cm = pd.DataFrame(val_cm, columns=["Pred 0", "Pred 1"], index=["0", "1"])
                    df_val_cm.to_csv(os.path.join(output_dir, "best_val_confusion_matrix.csv"), index=True)
                    with open(os.path.join(output_dir, "best_val_confusion_matrix.md"), "w") as f:
                        f.write(df_val_cm.to_markdown())

        # del running_loss, total_examples, res_dict, losses 
               
        # ----------

        if torch.device(device).type == "mps":
            gc.collect()
            torch.mps.empty_cache()
            # Force a sync so Metal finishes queued command buffers and frees driver memory
            torch.mps.synchronize()
            

        print()

    

        

    # Save final model
    torch.save(model.state_dict(), final_model_path)

        
            

    res = {}
    res["training_loss"] = train_losses
    res["model_selection_mode"] = model_selection_mode
    res["best_epoch"] = best_epoch
    
    if model_selection_mode == "standard":
        res["best_value"] = best_val_measure
    elif model_selection_mode == "accuracy_loss":
        res["best_value"] = best_val_acc
    elif model_selection_mode == "AdjScore":
        res["best_value"] = best_adj_score
    elif model_selection_mode == "GenGap":
        res["best_value"] = best_val_measure_within_gap if best_val_measure_within_gap != initial_val else best_val_measure

    res["best_model_path"] = best_model_path
    res["final_model_path"] = final_model_path
    res["last_epoch"] = epoch
    for k, v in train_results.items():
        res[k] = v

    return res




    

def make_predictions(model, dataloader, device, loss_fn=None, loss_fn_name=None):
    
    model = model.to(device)
    model.eval()

    outs = []
    out_labels = []
    preds = []
    formula_ids = []
    losses = []


    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):

            
            batch = batch.to(device)
            out = model(batch)
            
            labels = batch.label


            classification_method = model.classification_method


            match classification_method:    
                case "graph":
                    out = out.squeeze(-1)
                case "node":
                    out_Node_Level = out
                    out = out.squeeze(-1)
                    out = torch.sigmoid(out) # normalise in [0,1]
                    
                    
                    out = nodeLevelPrediction(out, batch, device)

                    myNodeOuts = out

                case _:        
                    print("learning.py: make_predictions 1: default classification case undefined")    
                    sys.exit(0)

            

            if loss_fn is not None:
                assert loss_fn_name is not None, "loss_fn_name must be provided if loss_fn is provided"
                match   loss_fn_name: 
                    case "mse":
                        lo = loss_fn(torch.sigmoid(out), labels).item()
                    case "bce":
                        lo = loss_fn(out, labels).item()                    
                    case "focal":
                        lo = loss_fn(out, labels).item()    
                    case "unsup2":
                        lo = loss_fn(out).item()                  
                    case _:                
                        pass
                losses.append(lo)

            match classification_method:    
                case "graph": # graph level classification
                    out = torch.sigmoid(out)
                    out = out.cpu().numpy().flatten()
                    pred_feasible = (out.squeeze() > 0.5).astype(int)
                    outs.extend(out)
                case "node": # node level classification
                    
                    out_Node_Level = torch.sigmoid(out_Node_Level)
                    
                    
                    pred_feasible = sat_predictions(out_Node_Level, batch, device).tolist()
                    
                    myNodeOuts = myNodeOuts.cpu().numpy().flatten()
                    outs.extend(myNodeOuts)



                case _:        
                    print("learning.py: make_predictions 2: default classification case undefined")      
                    sys.exit(0)

            preds.extend(pred_feasible)            
            labs = labels.cpu().numpy().astype(int)
            out_labels.extend(labs)
            
            
            formula_ids.extend(batch["formula_id"].cpu().numpy())

            del batch
                        

        


    res_dict = {
        "formula_id": formula_ids,
        "out": outs,
        "prediction": preds,
        "label": out_labels
    }

    return res_dict, losses        

def test_model(model, test_loader, device):
    
    model.eval()
    
    n_examples = len(test_loader.dataset)
    print(f"Testing on {n_examples} examples -- device: {device}")
    
    res_dict, _ = make_predictions(model, test_loader, device)
    df = pd.DataFrame.from_dict(res_dict)
    df["correct"] = df["prediction"] == df["label"]
    df["tp"] = (df["prediction"] == 1) & (df["label"] == 1)
    df["tn"] = (df["prediction"] == 0) & (df["label"] == 0)
    df["fp"] = (df["prediction"] == 1) & (df["label"] == 0)
    df["fn"] = (df["prediction"] == 0) & (df["label"] == 1)
    for col in ["correct", "tp", "tn", "fp", "fn"]:
        df[col] = df[col].astype(int)
    
    return df

from sklearn.metrics import accuracy_score, zero_one_loss, precision_score, recall_score, f1_score
def compute_metrics(df):
    metrics = {
        "accuracy": accuracy_score,
        "msr": zero_one_loss,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score
    }
    
    y_true = df["label"]
    y_pred = df["prediction"]

    
    cm = confusion_matrix(y_true, y_pred)

    res = {}
    for k, metric in metrics.items():
        res[k] = metric(y_true, y_pred)

    return res, cm
