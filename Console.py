from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')
tokenizer.padding_side = 'left'


# Let's chat for 4 lines
firstLine = True
historyOfDialog = ''
step = 0
while step < 4:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    userPhrase = tokenizer.encode(input("User:") + tokenizer.eos_token, return_tensors='pt')
    # print(userPhrase)

    # append the new user input tokens to the chat history
    botAnswer = torch.cat([historyOfDialog, userPhrase], dim=-1) if step > 0 else userPhrase
    firstLine = False

    # generated a response while limiting the total chat history to 1000 tokens,
    historyOfDialog = model.generate(
        botAnswer, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    # pretty print last ouput tokens from bot
    print("Tirion: {}".format(tokenizer.decode(historyOfDialog[:, botAnswer.shape[-1]:][0], skip_special_tokens=True)))

    step += 1

    if step == 3:
        firstLine = True
        botAnswer = ''
        step = 0
