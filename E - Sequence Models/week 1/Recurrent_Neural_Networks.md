## Course E - Sequence Models

## Week 1 - Recurrent Neural Networks

1. **Suppose your training examples are sentences (sequences of words). Which of the following refers to the jth word in the ith training example?**

- [x] x(i)\<j\>

- [ ] x\<i\>(j)

- [ ] x(j)\<i\>

- [ ] x\<\j\>(i)

```
Correct
We index into the i^{th}*i**t**h* row first to get the i^{th}*i**t**h* training example (represented by parentheses), then the j^{th}*j**t**h* column to get the j^{th}*j**t**h* word (represented by the brackets).
```

2. **Consider this RNN:**

   ![img](q2.png)

   **This specific type of architecture is appropriate when:**

- [x] Tx = Ty

- [ ] Tx < Ty

- [ ] Tx > Ty

- [ ] Tx = 1

```
Correct
It is appropriate when every input should be matched to an output.
```

3. **To which of these tasks would you apply a many-to-one RNN architecture? (Check all that apply).**

   ![img](q3.png)

- [ ] Speech recognition (input an audio clip and output a transcript)

- [x] Sentiment classification (input a piece of text and output a 0/1 to denote positive or negative sentiment)

```
Correct
Correct!
```

- [ ] Image classification (input an image and output a label)

- [x] Gender recognition from speech (input an audio clip and output a label indicating the speaker’s gender)

```
Correct
Correct!
```

4. **You are training this RNN language model.**

   ![img](q4.png)

   **At the t^{th} time step, what is the RNN doing? Choose the best answer.**

- [ ] Estimating P(y\<1>,y\<2>,…,y<t−1>)

- [ ] Estimating P(y^{\<t>})P(y\<t>)

- [x] Estimating P(y\<t>∣y\<1>,y\<2>,…,y\<t−1>)

- [ ] Estimating P(y\<t>∣y\<1>,y\<2>,…,y\<t>)

```
Correct
Yes, in a language model we try to predict the next step based on the knowledge of all prior steps.
```

5. **You have finished training a language model RNN and are using it to sample random sentences, as follows:**

   ![img](q5.png)

   **What are you doing at each time step t*t*?**

- [ ] (i) Use the probabilities output by the RNN to pick the highest probability word for that time-step as y^\<t>. (ii) Then pass the ground-truth word from the training set to the next time-step.

- [ ] (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as y^\<t>. (ii) Then pass the ground-truth word from the training set to the next time-step.

- [ ] (i) Use the probabilities output by the RNN to pick the highest probability word for that time-step as y^\<t>. (ii) Then pass this selected word to the next time-step.

- [x] (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as y^\<t>. (ii) Then pass this selected word to the next time-step.

```
Correct
Yes!
```

6. **You are training an RNN, and find that your weights and activations are all taking on the value of NaN (“Not a Number”). Which of these is the most likely cause of this problem?**

- [ ] Vanishing gradient problem.

- [x] Exploding gradient problem.

- [ ] ReLU activation function g(.) used to compute g(z), where z is too large.

- [ ] Sigmoid activation function g(.) used to compute g(z), where z is too large.

```
Correct
```

7. **Suppose you are training a LSTM. You have a 10000 word vocabulary, and are using an LSTM with 100-dimensional activations a\<t>. What is the dimension of Γu at each time step?**

- [ ] 1

- [x] 100

- [ ] 300

- [ ] 10000

```
Correct
Correct, Γ*u* is a vector of dimension equal to the number of hidden units in the LSTM.
```

8. **Here’re the update equations for the GRU.**

   ![img](q8.png)

   **Alice proposes to simplify the GRU by always removing the Γu. I.e., setting Γu = 1. Betty proposes to simplify the GRU by removing the Γr. I. e., setting Γr = 1 always. Which of these models is more likely to work without vanishing gradient problems even when trained on very long input sequences?**

- [ ] Alice’s model (removing Γ*u*), because if Γ*r*≈0 for a timestep, the gradient can propagate back through that timestep without much decay.

- [ ] Alice’s model (removing Γ*u*), because if Γ*r*≈1 for a timestep, the gradient can propagate back through that timestep without much decay.

- [x] Betty’s model (removing Γ*r*), because if Γ*u*≈0 for a timestep, the gradient can propagate back through that timestep without much decay.

- [ ] Betty’s model (removing Γ*r*), because if Γ*u*≈1 for a timestep, the gradient can propagate back through that timestep without much decay.

```
Correct
Yes. For the signal to backpropagate without vanishing, we need c^{<t>}*c*<*t*> to be highly dependant on c^{<t-1>}*c*<*t*−1>.
```

9. **Here are the equations for the GRU and the LSTM:**

   ![img](q9.png)

   **From these, we can see that the Update Gate and Forget Gate in the LSTM play a role similar to ___ and __ in the GRU. What should go in the the blanks?**

- [x] Γ*u* and 1−Γ*u*

- [ ] Γ*u* and Γ*r*

- [ ] 1−Γ*u* and Γ*u*

- [ ] Γ*r* and Γ*u*

```
Correct
Yes, correct!
```

10. **You have a pet dog whose mood is heavily dependent on the current and past few days’ weather. You’ve collected data for the past 365 days on the weather, which you represent as a sequence as *x*<1>,…,*x*<365>. You’ve also collected data on your dog’s mood, which you represent as *y*<1>,…,*y*<365>. You’d like to build a model to map from x*→*y. Should you use a Unidirectional RNN or Bidirectional RNN for this problem?**

- [ ] Bidirectional RNN, because this allows the prediction of mood on day t to take into account more information.

- [ ] Bidirectional RNN, because this allows backpropagation to compute more accurate gradients.

- [x] Unidirectional RNN, because the value of y^{\<t>}*y*<*t*> depends only on *x*<1>,…,*x*<*t*>, but not on *x*<*t*+1>,…,*x*<365>

- [ ] Unidirectional RNN, because the value of y^{\<t>}*y*<*t*> depends only on x^{\<t>}*x*<*t*>, and not other days’ weather.

```
Correct
Yes!
```

