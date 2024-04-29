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



from tqdm.notebook import tqdm
import datetime
import torch

def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, label_to_index, only_name, log = True):
    losses = []
    accuracies = []
    model = model.to(device)
    for i, epoch in enumerate(range(num_epochs)):
        for waveforms, sr, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            optimizer.zero_grad()
            waveforms = waveforms.to(device)
            outputs = model(waveforms.squeeze(1))

            # Convert string labels to integer indices
            target_indices = [label_to_index[label] for label in labels]

            # Convert the list of indices to a tensor
            target_tensor = torch.tensor(target_indices)

            # print(outputs['logits'])

            loss = criterion(outputs['logits'], target_tensor.to(device))
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}")
        torch.cuda.empty_cache()

        
        # predict on validation set
        val_losses = []
        val_accuracies = []
        for waveforms, sr, labels in tqdm(val_loader, total=len(val_loader), desc=f"Validation"):
            waveforms = waveforms.to(device)
            outputs = model(waveforms.squeeze(1))
            target_indices = [label_to_index[label] for label in labels]
            target_tensor = torch.tensor(target_indices)
            
            loss = criterion(outputs['logits'], target_tensor.to(device))
            # print(loss)
            val_losses.append(loss.item())
            val_accuracies.append((outputs['logits'].argmax(1) == target_tensor.to(device)).float())
        # break
        # print(val_losses)
        # print(val_accuracies)
        val_losses = [l for l in val_losses]
        val_accuracies = [a.item() for a in val_accuracies]

        loss = sum(val_losses)/len(val_losses)
        accuracy = sum(val_accuracies)/len(val_accuracies)

        losses.append(loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {loss}, Validation Accuracy: {accuracy}")
        torch.cuda.empty_cache()



    if log:
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{only_name}-{date}"
        os.makedirs(log_dir)
        # save acc and loss
        with open(f"{log_dir}/acc.txt", "w") as f:
            f.write(str(accuracies))
        with open(f"{log_dir}/loss.txt", "w") as f:
            f.write(str(losses))
    