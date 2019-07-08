# Tanh-Rescaling-Layer

By using tanh rescaling the output of the convolution, it rescales the scale of the output.  
And pass through the Softmax, the loss will become more sensitive if the class is hard to classify.

Due to the softmax fomula
![](https://github.com/ricky40403/Tanh-Rescaling-Layer/blob/master/softmax.png)  

By the characteristics of tanh, it will have drastic changes when the value is in the middle part.  

If we look at the probability, it means not classified well. So using can let the model get more sensitive when it is hard to classify.
Because the range of tanh is between -1 and 1, and it will cause the model hard to converge. So here use the tanh as the scaling scale.
So if value rescaling by tanh, it will not be constrained and get the behavior of tanh. 
![](https://github.com/ricky40403/Tanh-Rescaling-Layer/blob/master/tanh1.png)

By tanh and exponential, it will be more sensitive after rescaling if the value difference increase, .
![](https://github.com/ricky40403/Tanh-Rescaling-Layer/blob/master/tanh2.png)

Foward:  

y = (2 + tanh(x)) * x  

Backward:  
y' = (1 - tanh(x) ^2) + (2 + tanh(x))  

***
### prptotxt  

``` caffe
layer {
  name: "TanhScale"
  type: "TanhScale"
  bottom: "deconv1"
  top: "TanhScale"
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "TanhScale"
  bottom: "label"
  top: "loss"
  propagate_down: true
  propagate_down: false
  loss_param {
    weight_by_label_freqs: true
    class_weighting: 1.0
    class_weighting: 1.0
    class_weighting: 4.0
    class_weighting: 2.0
    class_weighting: 2.0
    class_weighting: 1.0
    class_weighting: 1.0
    normalization: VALID
  }
}
```
