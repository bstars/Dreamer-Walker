# Dreamer 

This is the final project for JHU 520.637 Fall 2023. I made a (readable) implementation of Dreamer in PyTorch. 
This work is insprired by 

- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- [adityabingi's implementation for Dreamer](https://github.com/adityabingi/Dreamer/tree/main)
- [abhayraw1's implementation for RSSM](https://github.com/abhayraw1/planet-torch)

The following is the result of my implementation (I sampled 1 episode every 100 iterations, and save the checkpoints every 1000 iterations).

<p>
  <img width="35%" src="https://github.com/bstars/Dreamer-Walker/blob/main/train_history.png">
</p>

In my code, I trained for 100k iterations, and the final behavior looks like    
<p>
  <img width="35%" src="https://github.com/bstars/Dreamer-Walker/blob/main/50.gif">
</p>

The model can also rollout a trajectory in "imagnination" (refer to Figure 5 in the paper), the following is my result,
the first row is "imagination" and the second row is the real trajectory.
<p>
  <img width="35%" src="https://github.com/bstars/Dreamer-Walker/blob/main/imagine.png">
</p>


The training time is 5 hours on a colab V100 GPU.
My code seems to learn slower than [adityabingi's implementation](https://github.com/adityabingi/Dreamer/tree/main).

I uploaded the ckeckpoint for the final model at ./ckpts/dreamer_50.pt, you can run it by (first uncomment the tester.sample_episode() in test.py)

```python
python test.py
```