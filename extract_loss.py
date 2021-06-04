import re
import sys

EPOCH_EXP = r"(?<=Train Epoch: )\d+"
LOSS_EXP = r"(?<=Loss: )\d+\.\d+"
EVAL_EPOCH_EXP = r"(?<=Eval Epoch: )\d+"
LR_EXP = r"(\d+\.\d+(e-\d+)?)(?=])"


def get_losses(infile):
    losses = []
    lrates = []
    eval_losses = [] 
    with open(infile, "r") as fh:
        line = fh.readline()

        while line:
            try:
                if "Train Epoch:" in line:
                    epoch = re.search(EPOCH_EXP, line).group(0)
                    loss = re.search(LOSS_EXP, line).group(0)
                    losses.append((epoch, loss))

                    lr_line = fh.readline()
                    lr = re.search(LR_EXP, lr_line).group(0)
                    lrates.append((epoch, lr))

                if "Eval Epoch:" in line:
                    epoch = re.search(EVAL_EPOCH_EXP, line).group(0)
                    loss = re.search(LOSS_EXP, line).group(0)
                    eval_losses.append((epoch, loss))
            
            except Exception:
                print(line)

            line = fh.readline()

    return losses, lrates, eval_losses


def prune_losses(losses):
    # only keep the last loss value for each epoch
    pruned_losses = []
    curr_epoch = 999
    losses.reverse()
    for epoch, loss in losses:
        if epoch != curr_epoch:
            curr_epoch = epoch
            pruned_losses.append((epoch, loss))

    pruned_losses.reverse()
    return pruned_losses



def save_to_csv(losses, filename):
    with open(filename, "w") as fh:
        for epoch, loss in losses:
            to_write = f"{epoch},{loss}\n"
            fh.write(to_write)


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    losses, lrates, eval_losses = get_losses(infile)
    losses = prune_losses(losses)
    lrates = prune_losses(lrates)
    save_to_csv(losses, f"{outfile}_losses.csv")
    save_to_csv(lrates, f"{outfile}_lrates.csv")
    save_to_csv(eval_losses, f"{outfile}_eval_losses.csv")
