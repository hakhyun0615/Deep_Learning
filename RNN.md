# TDNN(Time Delay Neural Network)
- goal: find temporal patterns with shift-invariance    
![](images/2023-04-20-15-18-28.png)
- network: MLP/FFNN(Feed Forward Neural Network)  
- data: sequential data(x_t, x_t-1, x_t-2, ...)  
- input: [x_t, x_t-1, x_t-2, x_t-3], [x_t-1, x_t-2, x_t-3, x_t-4], ... (sliding a window of size n across sequential data)  
![](images/![](2023-04-20-15-14-42.png).png)  
- disadvantage:
    - success depends on appropriate window size
        - small window size does not capture
        - large window increases the parameter number and may add unnecessary noise
    - works well for short-memory problems but not for long-memory problems
    - fixed window size cannot handle sequential data of variable length such as language translation
    - FFNNs do not have any memory of past data(only able to capture temporal dependency within the window size)
    - FFNNs treat input data as a multidimensional feature vector rather than sequence of observations(lose the benefit of sequential information)

# RNN(Recurrent Neural Network)
![](images/2023-05-02-13-59-41.png)
- goal: make predictions based on current input and previous inputs(FFNN makes predictions based on only the current input)  
![](images/2023-04-20-15-47-49.png)
- input: 3D tensor with shape (batch_size(number of data for 1 iteration(weight update)), timesteps(of Truncated BPTT), input_dim(of x_t))
- forward propagation
    - input: x_t
    - prior hidden state: h_t-1(selective memory from previous inputs)
    - current hidden state: h_t
        - weights: W_xh, W_hh(shared at every time step)
        - bias: b_h
        - activation function: tanh(-1~1->reflection of past memory)
    - output: y_t
        - weight: W_hy(shared at every time step)
        - bias: b_y
        - activation function: depends  
    ![](images/2023-04-21-16-16-19.png)
- BPTT(Back Propagation Through Time)
    - goal: update U(W_xh), V(W_hh), W(W_hy)
    - weight update process
        - compute all weights in the unfolded network(U_0, U_1, ... & V_1, v_2, ...)
        - U = Avg/Sum(U_0, U_1, ...), V = Avg/Sum(V_1, v_2, ...)
    ![](images/2023-04-21-16-40-11.png)
- Truncated BPTT(Truncated Back Propagation Through Time)
    - problem
        - BPTT can be slow when dealing with long sequential data
        - multiplication of gradients over many timesteps can lead to gradient vanishing/exploding problem
    - solve: backpropagation done over each subsequences
    - hyperparameter
        - number of timesteps(lookback)
            - divide long input sequence into overlapping subsequences with same time interval
            - should be long enough to capture relevant past information and should be short enough to train efficiently
            - example
                - batch size: 5
                - timestep: 5
                - initial hidden state
                    - Default mode: random
                    - Stateful mode: stateful = true(better than random)
                        - if previous hidden state exits from other timesteps, use it
                        - should not shuffle within mini-batch(sequence in each mini-batch matters)
                        - no remainder for numbers of total samples/batch size
                        - example
                            - first time step(x_0, x_1, x_2, x_3, x_4) in first mini-batch makes h_4
                            - h_4 is used as an initial hidden state for first time step(x_5, x_6, x_7, x_8, x_9) in second mini-batch
                - weight update: Avg/Sum(Avg/Sum(gradients) of each timesteps) for each mini-batch
                    - can shuffle within mini-batchs(each is independant)
                    - cannot shuffle within timesteps(each is relevant)