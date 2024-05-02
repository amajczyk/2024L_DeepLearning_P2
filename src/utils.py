import os
import re
import hashlib

# -------------------- Function from the competition's files --------------------
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename:str, validation_percentage:float, testing_percentage:float) -> str:
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
        filename: File path of the data sample.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

    Returns:
        String, one of 'training', 'validation', or 'testing'.
    """

# ------------------- My addition to check the percentages -------------------
    # Check that the percentages do not exceed 100%
    if validation_percentage + testing_percentage >= 1:
        raise ValueError('validation_percentage + testing_percentage must be less than 100')
    
    # if they are more than 50%, raise a warning
    if validation_percentage > .5 or testing_percentage > .5 or validation_percentage + testing_percentage > .5:
        print("WARNING: High validation and/or testing percentage. It is recommended to use a lower value.")
# ------------------- End of my addition -------------------

        

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name = hash_name.encode('utf-8')
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                        (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

import json


from tqdm.notebook import tqdm
import datetime
import torch
import json
import json

def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, label_to_index, only_name, log = True, description = "", lstm = False):
    losses = []
    accuracies = []
    model = model.to(device)
    train_losses = []
    for i, epoch in enumerate(range(num_epochs)):
        epoch_losses = []
        for waveforms, sr, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            optimizer.zero_grad()
            waveforms = waveforms.to(device)
        
            outputs = model(waveforms.squeeze(1))
            

            # Convert string labels to integer indices
            target_indices = [label_to_index[label] for label in labels]

            # Convert the list of indices to a tensor
            target_tensor = torch.tensor(target_indices)

            # print(f"output logits: {outputs}")
            if lstm:
                outputs = outputs
            else:

                outputs = outputs['logits']
            loss = criterion(outputs, target_tensor.to(device))
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        

        loss = sum(epoch_losses)/len(epoch_losses)
        train_losses.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss}")
        torch.cuda.empty_cache()

        
        # predict on validation set
        val_losses = []
        val_accuracies = []
        for waveforms, sr, labels in tqdm(val_loader, total=len(val_loader), desc=f"Validation"):
            waveforms = waveforms.to(device)
            outputs = model(waveforms.squeeze(1))
            target_indices = [label_to_index[label] for label in labels]
            target_tensor = torch.tensor(target_indices)
            
            if lstm:
                outputs = outputs
            else:
                outputs = outputs['logits']
            # print(loss)
            loss = criterion(outputs, target_tensor.to(device))
            val_losses.append(loss.item())
            val_accuracies.append((outputs.argmax(1) == target_tensor.to(device)).float())
        # break
        # print(val_losses)
        # print(val_accuracies)
        val_losses = [l for l in val_losses]
        # print(val_accuracies)
        val_len = [len(a) for a in val_accuracies]
        val_accuracies = [a.sum().item() for a in val_accuracies]
        
        # print(val_accuracies, val_len)
        

        loss = sum(val_losses)/len(val_losses)
        accuracy = sum(val_accuracies)/sum(val_len)

        losses.append(loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {loss}, Validation Accuracy: {accuracy}")
        torch.cuda.empty_cache()



    if log:
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{only_name}-{date}"
        if description != "":
            log_dir += f"_{description}"
        os.makedirs(log_dir)
        # save acc and loss
        data = {
            "val_acc": accuracies,
            "val_loss": losses,
            "train_loss": train_losses
        }

        with open(f"{log_dir}/data.json", "w") as f:
            json.dump(data, f)

        return log_dir
    

def test(model, test_loader, criterion, device, label_to_index, only_name, log_dir, description = "", log = True, lstm = False):
    model.eval()
    losses = []
    accuracies = []
    predictions = []
    real_labels = []
    for waveforms, sr, labels in tqdm(test_loader, total=len(test_loader), desc="Testing"):
        waveforms = waveforms.to(device)
        outputs = model(waveforms.squeeze(1))
        target_indices = [label_to_index[label] for label in labels]
        target_tensor = torch.tensor(target_indices)
        if lstm:
            outputs = outputs
        else:
            outputs = outputs['logits']
        loss = criterion(outputs, target_tensor.to(device))
        losses.append(loss.item())
        accuracies.append((outputs.argmax(1) == target_tensor.to(device)).float())
        predictions.extend(outputs.argmax(1).cpu().numpy())
        real_labels.extend(target_tensor.cpu().numpy())
    
    lens = [len(a) for a in accuracies]
    accuracies = [a.sum().item() for a in accuracies]
    
    # print(val_accuracies, val_len)


    

    loss = sum(losses)/len(losses)
    accuracy = sum(accuracies)/sum(lens)

    torch.cuda.empty_cache()

    


    if log:
        if description != "":
            log_dir += f"_{description}"
       
        try:    
            data = json.load(open(f"{log_dir}/data.json", "r"))
        except:
            pass
        # save acc and loss

        predictions = [int(p) for p in predictions]
        real_labels = [int(p) for p in real_labels]
        
        
        data["test_correct_in_batch"] = accuracies
        data["test_losses"] = losses
        data["test_loss"] = loss
        data["test_batch_lens"] = lens
        data["test_accuracy"] = accuracy
        data["predictions"] = predictions
        data["real_labels"] = real_labels
        data["label_to_index"] = label_to_index

        print(data)
        

        with open(f"{log_dir}/data.json", "w") as f:
            json.dump(data, f)


    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return accuracy