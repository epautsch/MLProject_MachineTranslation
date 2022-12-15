from __future__ import unicode_literals, print_function, division
import random
import DataPrep as dp
import Networks as net
import torch
import Training as trn
import Evaluate as eval
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
input_lang, output_lang, pairs = dp.prepareData('eng', 'spa', True)

try:
    # encoder1 = torch.load('encoder_gru_6L_tf0_2.pth', map_location=torch.device('cpu'))
    # attn_decoder1 = torch.load('attn_decoder1_gru_6L_tf0_2.pth', map_location=torch.device('cpu'))
    encoder1 = torch.load('encoder_gru_6L_tf0_2.pth')
    attn_decoder1 = torch.load('attn_decoder1_gru_6L_tf0_2.pth')
    # eval.evaluateAndShowAttention('Cada vez que escucho esta canción, lloro.', encoder1, attn_decoder1, input_lang, output_lang)
    user_input = input('Input sentence to be translated: ')
    while user_input.lower() != 'exit':
        try:
            output_words, attentions = eval.evaluate(encoder1, attn_decoder1, user_input, input_lang, output_lang)
            output_sentence = ' '.join(output_words)
            print(output_sentence)
            user_input = input('Input sentence to be translated: ')
        except KeyError:
            print("You entered a word not in the dictionary. String cannot be processed.")
            user_input = input('Input sentence to be translated: ')

except FileNotFoundError:

    hidden_size = 256
    encoder1 = net.EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = net.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trn.trainIters(encoder1, attn_decoder1, 450000, pairs, input_lang, output_lang, print_every=1000)
    print(random.choice(pairs))
    eval.evaluateRandomly(encoder1, attn_decoder1, pairs, input_lang, output_lang)
    torch.save(encoder1, 'encoder_gru_6L_tf0_2.pth')
    torch.save(attn_decoder1, 'attn_decoder1_gru_6L_tf0_2.pth')
    output_words, attentions = eval.evaluate(encoder1, attn_decoder1, "Mi nombre es .", input_lang, output_lang)
    plt.matshow(attentions.numpy())

    # eval.evaluateAndShowAttention('Los asiáticos generalmente tienen pelo oscuro.', encoder1, attn_decoder1, input_lang, output_lang)

    eval.evaluateAndShowAttention('Cada vez que escucho esta canción, lloro.', encoder1, attn_decoder1, input_lang, output_lang)

    eval.evaluateAndShowAttention('Me las apañé para arreglar mi coche yo mismo.', encoder1, attn_decoder1, input_lang, output_lang)

    eval.evaluateAndShowAttention('Quiero que mañana vayas en avión a Boston.', encoder1, attn_decoder1, input_lang, output_lang)



