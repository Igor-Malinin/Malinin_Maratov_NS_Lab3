from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
import torch
import discord
from discord.ext import commands
import os
import warnings
warnings.filterwarnings("ignore")

client = discord.Client(intents = discord.Intents().all())


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelWithLMHead.from_pretrained('output-small')
    tokenizer.padding_side = 'left'
    historyOfDialog = ''
    firstLine = True
    if message.author == client.user:
        return

    userPhrase = tokenizer.encode(str(message.content) + tokenizer.eos_token, return_tensors='pt')

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

    await message.channel.send(tokenizer.decode(historyOfDialog[:, botAnswer.shape[-1]:][0], skip_special_tokens=True))


client.run('MTA1NTE2Mzc3MzU1NzY4MjI2Ng.Gn72YE.6HoIaQGBBBJRdYAqSCtmG6euFStKMQgEMXczqc')
