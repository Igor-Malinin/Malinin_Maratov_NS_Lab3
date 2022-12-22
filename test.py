from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')
tokenizer.padding_side = 'left'

firstLine = True
historyOfDialog = ''
while(True):
    userPhrase = tokenizer.encode(input("User:") + tokenizer.eos_token, return_tensors='pt')

    botAnswer = torch.cat([historyOfDialog, userPhrase], dim=-1) if not firstLine else userPhrase

    historyOfDialog = model.generate(
        botAnswer, max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    print("Tirion: {}".format(tokenizer.decode(historyOfDialog[:, botAnswer.shape[-1]:][0], skip_special_tokens=True)))