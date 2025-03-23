import torch
from llama import load_pretrained

# Set seeds for reproducibility
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Move computations to CPU for consistency
device = torch.device("cpu")

sanity_data = torch.load("./sanity_check.data", map_location=device)
sent_ids = torch.tensor([
    [101, 7592, 2088, 102, 0, 0, 0, 0],
    [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]
], device=device)

# Load the model and move it to CPU
llama = load_pretrained("E:\\Projects\\AI_Assignments\\ASM1-Development-LLM-model-main\\data\\stories42M.pt").to(device)

with torch.no_grad():
    logits, hidden_states = llama(sent_ids)
    assert torch.allclose(logits, sanity_data["logits"], atol=1e-4, rtol=1e-3), "Logits do not match."
    assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-4,
                          rtol=1e-3), "Hidden states do not match."
    print("Your Llama implementation is correct!")
