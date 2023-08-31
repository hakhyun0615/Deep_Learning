# Seq2Seq limitation
![Alt text](<images/image-11.png>)  
- having context as a fixed size vector can cause a bottleneck and performance degradation
![Alt text](<images/스크린샷 2023-08-31 오전 10.21.07.png>)
- although the decoder can improve performance by consulting the context vector each time(reduce information loss), it still requires compressing the sentences into a single vector so bottleneck still exists
- since modern GPUs have a lot of memory and are capable of fast parallel processing, wouldn't it be possible to solve this problem by receiving all the output from the source sentence as input each time?

# Seq2Seq with Attention
![Alt text](<images/스크린샷 2023-08-31 오전 10.40.26.png>)
- save the hidden states from the source sentence separately and refer to them each time a word is generated in the decoder
![Alt text](<images/스크린샷 2023-08-31 오전 10.46.28.png>)
- Energy
    - every time the ith index of the decoder produces an output word, it considers all jth hidden state of encoders
    - able to find which h value has the greatest correlation
- Weight
    - softmax(energy)
    - the probability of correlation for each value of h
![Alt text](<images/스크린샷 2023-08-31 오전 10.52.39.png>)
- use weighted sum for weight and encoder hidden states as an another input of decoder
![Alt text](<images/스크린샷 2023-08-31 오전 10.52.39-1.png>)
- visually check which input from the encoder most referenced for each word output from the decoder through the attention weight

# Transformer
![Alt text](<images/스크린샷 2023-08-31 오전 11.02.14.png>)
- do not use RNN or CNN at all → use Positional Encoding instead
![Alt text](<images/스크린샷 2023-08-31 오후 4.55.51.png>)
- the output of the last encoder layer is input to all decoder layers and used in the second attention
## Encoder
![Alt text](<images/스크린샷 2023-08-31 오후 4.51.19.png>)
- Embedding → Attention
![Alt text](<images/스크린샷 2023-08-31 오전 11.04.13.png>)
- Traditional Embedding used in RNN
![Alt text](<images/스크린샷 2023-08-31 오전 11.05.46.png>)
- Embedding used in Transformer
![Alt text](<images/스크린샷 2023-08-31 오전 11.08.01.png>)
- Encoder Self-Attention
    ![Alt text](<images/스크린샷 2023-08-31 오후 5.47.27.png>)
    - finds how each word is related to each other
![Alt text](<images/스크린샷 2023-08-31 오전 11.11.20.png>)
- Residual Learning
## Decoder
![Alt text](<images/스크린샷 2023-08-31 오후 4.54.22.png>)
- Masked Decoder Self-Attention
    ![Alt text](<images/스크린샷 2023-08-31 오후 5.49.38.png>)
    - refer only to the preceding words
- Encoder-Decoder Attention
    ![Alt text](<images/스크린샷 2023-08-31 오후 5.49.57.png>)
    - Decoder: Query
    - Encoder: Key, Value
## Positional Encoding
![Alt text](<images/스크린샷 2023-08-31 오후 5.56.57.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.57.41.png>)
## Multi-Head Attention layer
![Alt text](<images/스크린샷 2023-08-31 오후 5.09.29.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.13.45.png>)
- Attention: finding out how a word is related to other words(a sentence)
- Q(Query): a word
- K(Key): a sentence
- V(Value)
### Example
![Alt text](<images/스크린샷 2023-08-31 오후 5.22.14.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.31.21.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.31.49.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.32.59.png>)
![Alt text](<images/스크린샷 2023-08-31 오후 5.37.23.png>)
- for each head, a vector of the same dimensions as the input Query, Key, and Value is created

![Alt text](<images/스크린샷 2023-08-31 오후 5.39.58.png>)
- if you concat all the results from each head, it becomes the same as the embedding dimension again, therefore the dimensions remain the same even after performing MultiHead(Q,K,V)