import torch
from braintumorref import CNN
model = CNN()
model.load_state_dict(torch.load('weights.pt'))

model = model.to(device) # Set model to gpu
model.eval();

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("Who are you?")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Ishan')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

print("heloo")