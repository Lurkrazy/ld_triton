ContinuousPipe: eliminate bubbles in the pipe parallel completely

# Abstract
Pipe parallel in traning deep neural network has some challenges. One of them is that, traditional pipe parallel always has bubbles of a fix ratio, for example, Gpip and PipStream. The essential problem of traditional pipe parallel has two stages. Firstly, split a mini_batch  into smaller micro_baches. Secondly, schedule the micro_batches in one mini_batch step. This method leads to the fix ratio of bubbles. In this paper, we give up the two-stage process, adapt one-stage process, via, schedule the mini_batch directly. This method will only generate bubbles with fixed numbers. With the training steps increasing, because the bubbles has fixed numbers, the ratio of bubbles will approach to zero.

# Introduction
Pipe parallel has many advantages compared to other parallel methods,for examle, data parallel and tensor parallel. 

Data parallel needs additional parameters backup, for example, dp is four, then it needs four backups of the parameters, this will multiplied increase memory demand.

Tensor parallel needs additional communications. If we split the tensor not correctly, the communications will become major issues,that is the time of the communicaitons will occupy much trainging time, then will lead GPU idel. 

But, pipe parrallel also has some limits.First, it could not split too depth, because it needs cache the activations in the boundary, when the model is too big, for example, recently most llm large models, after using pipe parallel, the memory still is not enough beacuse the depth is small, then single gpu haved partial parameters(total_parameters/depth) still has a big size, which leads that the memory single gpu is not enough. In the engineering practice, it will cause oom. Second, traditional pipe parallel methods, for example, Gpip and PipStream, always have bubbles with a fix ratio, they could not utilize the gpu completely,that is they always have some gpus idel.

This paper focuses on the second problem. The compeletly diffrent method we use will eliminate the bubbles which are always in the tradition pipe parallel. We give up the two-stage process in traditional pipe parallel, adapt one-stage process, schedule the mini_batch directly.

# Background &Related Work
In this section, we always assume num_stage is 4 and num_micro_batch is 8
## Gpipe 
Gpipe is the firt pipe parallel which introduces the two-stage method, two-stage method first splits a mini-batch of training examples into smaller micro-batches, then pipelines the execution of each set of micro-batches over cells.
Its execute order is this:
|   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |  21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  0F |  1F |  2F |  3F |  4F |  5F |  6F |  7F |  *  |  *  |  *  |  *  |  *  |  *  |  7B |  6B |  5B |  4B |  3B |  2B |  1B |  0B |
|  *  |  0F |  1F |  2F |  3F |  4F |  5F |  6F |  7F |  *  |  *  |  *  |  *  |  7B |  6B |  5B |  4B |  3B |  2B |  1B |  0B |  *  |
|  *  |  *  |  0F |  1F |  2F |  3F |  4F |  5F |  6F |  7F |  *  |  *  |  7B |  6B |  5B |  4B |  3B |  2B |  1B |  0B |  *  |  *  |
|  *  |  *  |  *  |  0F |  1F |  2F |  3F |  4F |  5F |  6F |  7F |  7B |  6B |  5B |  4B |  3B |  2B |  1B |  0B |  *  |  *  |  *  |

And Gpipe have some bubbles, which are represented by i$*$ in the table.

Let's begin deriving the formula of $bubble\_ratio$ . For simple, we assume
stages corresponding one-to-one with gpus and layers of model. 

$mini\_batch\_size = micro\_batch\_size * num\_micro\_batch$

$total\_computes= 2 * (num\_micro\_batch + num\_stage - 1) * num\_stage$

$valid\_computes = 2 * num\_micro\_batch * num\_stage$

$total\_bubbles = total\_computes - valid\_computes = 2 * (num\_stage - 1) * num\_stage$

$bubble\_ratio = \frac{total\_bubbles}{total\_computes}$

$= \frac{2 * (num\_stage - 1) * num\_stage}{2 * (num\_micro\_batch + num\_stage - 1) * num\_stage}$

$= \frac{num\_stage - 1}{num\_micro\_batch + num\_stage - 1}$

When $num\_stage = 1$, $bubble\_ratio=0$

$= \frac{1}{\frac{num\_micro\_batch}{num\_stage - 1} + 1}$

When $num\_stage > 1$, $bubble\_ratio$ is the decreasing function of $num\_micro\_batch$

Under the framework provided by Gpipe, when mini_batch_size is fixed, num_micro_batch has upper limit. When micro_batche_size=1, the maximum of num_micro_batch is mini_batch_size and the minimum of  $bubble\_ratio$ is $\frac{1}{\frac{mini\_batch\_size}{stages - 1} + 1}$

## PipeDream
 PipeDream schedules the forward passes and backward passes in diffrent orders.

|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  | 13  | 14  | 15  | 16  | 17  | 18  | 19  | 20  | 21  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  0F |  *  | 1F  |  *  | 2F  |  *  | 3F  | 0B  | 4F  | 1B  | 5F  | 2B  | 6F  | 3B  | 7F  | 4B  |  *  | 5B  |  *  | 6B  |  *  | 7B  |
|  *  | 0F  |  *  | 1F  |  *  | 2F  | 0B  | 3F  | 1B  | 4F  | 2B  | 5F  | 3B  | 6F  | 4B  | 7F  | 5B  |  *  | 6B  |  *  | 7B  |  *  |
|  *  |  *  | 0F  |  *  | 1F  | 0B  | 2F  | 1B  | 3F  | 2B  | 4F  | 3B  | 5F  | 4B  | 6F  | 5B  | 7F  | 6B  |  *  | 7B  |  *  |  *  |
|  *  |  *  |  *  | 0F  | 0B  | 1F  | 1B  | 2F  | 2B  | 3F  | 3B  | 4F  | 4B  | 5F  | 5B  | 6F  | 6B  | 7F  | 7B  |  *  |  *  |  *  |

Above the table, the bubble_ratio of PipeDream is same as the the bubble_ratio of GPipe

$bubble\_ratio = \frac{1}{\frac{mini\_batch\_size}{stages - 1} + 1}$

For $bubble\_ratio->0$ ,We have two view to look at the PipeDream. 

First view, let $num\_micro\_batch -> ∞$ directly, then $bubble\_ratio->0$ . We already know, this is not possible in the engineering practice.

Second view,  that is the view in our paper, via ContinuousPipe. We will introduce Parallel Training in ContinuousPipe in the next section.

# Parallel Training in ContinuousPipe

Now, let us obsverse carefully. Under the framework provided by PipeDream, the execution order is $0F->0B->1F->1B->2F->2B->...$ ，it is equivalent to the ordinary backpropagation algorithm of batch_size taking micro_batch_size.

In this case, let us think differently, Why don't we just schedule directly multiple mini_batches. The execution order is
$0MF->0MB->1MF->1MB->2MF->2MB->...$ , MF represent mini_batch forward pass and MB represent mini_batch backword pass.

Similar to GPipe, Let's begin deriving the formula of $bubble\_ratio$ 

$total\_computes= (num\_mini\_batch + 2 * (num\_stage - 1)) * num\_stage$

$valid\_computes = num\_micro\_batch * num\_stage$

$total\_bubbles = total\_computes - valid\_computes = 2 * (num\_stage - 1) * num\_stage$

$bubble\_ratio = \frac{total\_bubbles}{total\_computes}$

$=\frac{2 * (num\_stage - 1) * num\_stage}{(num\_mini\_batch + 2 * (num\_stage - 1)) * num\_stage}$

$=\frac{1}{\frac{num\_mini\_batch}{2 * (num\_stage - 1)} + 1}$

$num\_mini\_batch = \frac{num\_data * num\_epoch}{mini\_batch\_size}$

We know, if the time is enough, $num\_epoch -> ∞$, so $num\_mini\_batch -> ∞$ ,this is possible in the engineering practice, because we can set the $mini\_batch\_size = 1$, the $num\_mini\_batch=num\_data * num\_epoch$ is very large, the $bubble\_ratio$ will be very small. 

|   0  |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  10  |  11  |  12  |  13  |  14  |  15  |  16  |  17  |  18  |  19  |  20  |  21  |  22  |  23  |  24  |  25  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  0MF |  *   |  1MF |  *   |  2MF |  *   |  3MF |  0MB |  4MF |  1MB |  5MF |  2MB |  6MF |  3MB |  7MF |  4MB |  8MF |  5MB |  9MF |  6MB | 10MF |  7MB | 11MF |  8MB | 12MF |  9MB |
|   *  |  0MF |  *   |  1MF |  *   |  2MF |  0MB |  3MF |  1MB |  4MF |  2MB |  5MF |  3MB |  6MF |  4MB |  7MF |  5MB |  8MF |  6MB |  9MF |  7MB | 10MF |  8MB | 11MF |  9MB | 12MF |
|   *  |  *   |  0MF |  *   |  1MF |  0MB |  2MF |  1MB |  3MF |  2MB |  4MF |  3MB |  5MF |  4MB |  6MF |  5MB |  7MF |  6MB |  8MF |  7MB |  9MF |  8MB | 10MF |  9MB | 11MF | 10MB |
|   *  |  *   |  *   |  0MF |  0MB |  1MF |  1MB |  2MF |  2MB |  3MF |  3MB |  4MF |  4MB |  5MF |  5MB |  6MF |  6MB |  7MF |  7MB |  8MF |  8MB |  9MF |  9MB | 10MF | 10MB | 11MF |
# Evaluation
# Conclusion
In this paper, we give two diffrent view to eliminate the bubble_ratio. The first is PipeDream view, it needs $mini_batch_size  -> ∞$ this is not possible in the engineering practice.Second is our ContinuousPipe, it needs $num\_data * num\_epoch -> ∞$, this is not possible in the engineering practice. In practice in llm large model, $num\_data$ is always large enough, bubble_ratio will always smaller than PipeDream's 
# References

GPipe: Easy Scaling with Micro-Batch Pipeline
 Parallelism

PipeDream: Fast and Efficient Pipeline Parallel DNN Training

Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

https://github.com/deepspeedai/DeepSpeed

https://siboehm.com/articles/22/pipeline-parallel-training

https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md

