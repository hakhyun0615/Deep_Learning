# Seq2Seq limitation
- having context as a fixed size vector can cause a bottleneck and performance degradation
![Alt text](<images/image-11.png>)  
- although the decoder can improve performance by consulting the context vector each time(reduce information loss), it still requires compressing the sentences into a single vector so bottleneck still exists
![Alt text](<images/image-12.png>)  
- since modern GPUs have a lot of memory and are capable of fast parallel processing, wouldn't it be possible to solve this problem by receiving all the output from the source sentence as input each time?

# Seq2Seq with Attention
- save the hidden states from the source sentence separately and refer to them each time a word is generated in the decoder
![Alt text](<images/image-13.png>)

- Energy and Weight
    - ![Alt text](<images/image-14.png>)
    - Energy
        - every time the ith index of the decoder produces an output word, it considers all jth hidden state of encoders
        - able to find which h value has the greatest correlation
    - Weight
        - softmax(energy)
        - the probability of correlation for each value of h

- use weighted sum for weight and encoder hidden states as an another input of decoder
![Alt text](<images/image-15.png>)

- visually check which input from the encoder most referenced for each word output from the decoder through the attention weight
![Alt text](<images/image-37.png>)

# Transformer
![Alt text](<images/image-16.png>)
- do not use RNN or CNN at all → use Positional Encoding instead

- the output of the last encoder layer is input to all decoder layers and used in the second attention
![Alt text](<images/image-17.png>)
## Positional Encoding
![Alt text](<images/image-27.png>)
![Alt text](<images/image-28.png>)
## Encoder
![Alt text](<images/image-18.png>)
- Embedding → Attention

- Traditional Embedding used in RNN
![Alt text](<images/image-19.png>)

- Embedding used in Transformer
![Alt text](<images/image-20.png>)

- Encoder Self-Attention
    - finds how each word is related to each other
    ![Alt text](<images/image-21.png>)
    ![Alt text](<images/image-22.png>)
    
- Residual Learning
![Alt text](<images/image-23.png>)
## Decoder
![Alt text](<images/image-24.png>)
- Masked Decoder Self-Attention
    - refer only to the preceding words
    ![Alt text](<images/image-25.png>)
- Encoder-Decoder Attention
    - Decoder: Query
    - Encoder: Key, Value
    ![Alt text](<images/image-26.png>)

## Multi-Head Attention layer
- Attention: finding out how a word is related to other words(a sentence)
- Q(Query): a word
- K(Key): a sentence
- V(Value)
- ![Alt text](<images/image-29.png>)
- ![Alt text](<images/image-30.png>)

### Example
![Alt text](<images/image-31.png>)
![Alt text](<images/image-32.png>)
![Alt text](<images/image-33.png>)
![Alt text](<images/image-34.png>)
![Alt text](<images/image-35.png>)
- for each head, a vector of the same dimensions as the input Query, Key, and Value is created

![Alt text](<images/image-36.png>)
- if you concat all the results from each head, it becomes the same as the embedding dimension again, therefore the dimensions remain the same even after performing MultiHead(Q,K,V)