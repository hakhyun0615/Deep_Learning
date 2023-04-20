# TDNN(Time Delay Neural Network)
network: FFNN(MLP)
data: sequential data(x_t, x_t-1, x_t-2, ...)
input: [x_t, x_t-1, x_t-2, x_t-3], [x_t-1, x_t-2, x_t-3, x_t-4], ... (sliding a window of size n across sequential data)
![](images/![](2023-04-20-15-14-42.png).png)
goal: find temporal patterns with shift-invariance
![](images/2023-04-20-15-18-28.png)
disadvantage:
- success depends on appropriate window size
    - small window size does not capture
