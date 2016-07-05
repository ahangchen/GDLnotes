# 理解LSTM 网络
* Posted on August 27, 2015 *

## 循环神经网络

人不会每时每刻都从抓取信息这一步开始思考。你在读这篇文章的时候，你对每个次的理解是基于你对以前的词汇的理解的。你不会把所有东西都释放出来然后再从抓取信息开始重新思考，你的思维是有持续性的。

传统的神经网络不能做到这一点， 而且好像这是传统神经网络的一个主要缺点。例如，想象你想要区分一个电影里的每个时刻正在发生的事情。一个传统的神经网络将会如何利用它对过去电影中事件的推理，来预测后续的事件，这个过程是不清晰的。

循环神经网络解决了这个问题。在循环神经网络里，有循环，允许信息持续产生作用。

![](../../res/RNN-rolled.png)

循环神经网络有循环

在上面的图中，一大块神经网络，A，观察一些输入x<sub>t</sub>，输出一个值h<sub>t</sub>。循环允许信息从网络的一步传到下一步。

这些循环使得循环神经网络似乎有点神秘。然而，如果你想多一点，其实它们跟一个正常的神经网络没有神秘区别。一个循环神经网络可以被认为是同一个网络的多重副本，每个部分会向继任者传递一个信息。想一想，如果我们展开了循环会发生什么：

![](../../res/RNN-unrolled.png)

An unrolled recurrent neural network.

这个链式本质揭示了，循环神经网络跟序列和列表是紧密相关的。它们是神经网络为这类数据而生的自然架构。

并且它们真的起了作用！在过去的几年里，应用RNN到许多问题中都取得了难以置信的成功：语音识别，语言建模，翻译，图像截取，等等。我会留一个话题，讨论学习Andrej Karpathy的博客能够取得多么令人惊艳的成绩：

![The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。但它们真的相当惊艳。

与这些成功紧密相关的是对LSTM的使用，一个非常特殊的循环神经网络的类型。它在许多任务上都能比标准的RNN工作的好得多。几乎所有基于RNN的神经网络取得的激动人心的成果都由LSTM获得。这篇文章将要探索的就是这些LSTM。

## Long-Term依赖问题

RNN吸引人的一个地方是它们能够链接先前的信息与当前的任务，比如使用先前的视频帧可能预测对于当前帧的理解。如果RNN能够做到这种事情，它们会变得极度有用。但真的可以吗？不好说。

有时候，我们只需要查看最近的信息来执行现在的任务，例如，考虑一个语言模型试图基于先前的词预测下一个词。如果我们要预测“the clouds are in the sky”，我们不需要其他更遥远的上下文 —— 非常明显，下一个词就应该是sky。在这样的例子中，相关信息和目的地之间的距离是很小的。RNN可以学着区使用过去的信息。

![](../../res/RNN-shorttermdepdencies.png)

但也有一些情况是我们需要更多上下文的。考虑预测这个句子中最后一个词：“I grew up in France... I speak fluent French.” 最近的信息表明下一个词可能是一种语言的名字，但如果我们想要找出是哪种语言，我们需要从更久远的地方获取France的上下文。相关信息和目标之间的距离完全可能是非常巨大的。

不幸的是，随着距离的增大，RNN变得不能够连接信息。

![](../../res/RNN-longtermdependencies.png)

长期依赖导致的神经网络困境

理论上，RNN是绝对能够处理这样的“长期依赖的”。人类可以仔细地从这些词中找到参数然后解决这种形式的一些雏形问题。然而，实践中，RNN似乎不能够学习到这些。 Hochreiter (1991) [German] 和 Bengio, et al. 1994年曾探索过这个问题，他们发现了一些非常根本的导致RNN难以生效的原因。

万幸的是，LSTM没有这个问题！


## LSTM 网络

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![](../../res/LSTM3-SimpleRNN.png)

The repeating module in a standard RNN contains a single layer.

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![](../../res/LSTM3-chain.png)

The repeating module in an LSTM contains four interacting layers.

Don’t worry about the details of what’s going on. We’ll walk through the LSTM diagram step by step later. For now, let’s just try to get comfortable with the notation we’ll be using.

![](../../res/LSTM2-notation.png)

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

## LSTM背后的核心思想

LSTM的关键在于cell的状态，也就是图中贯穿顶部的那条水平线。

cell的状态像是一条传送带，它贯穿整条链，其中只发生一些小的线性作用。信息流过这条线而不改变是非常容易的。

![](../../res/LSTM3-C-line.png)

LSTM确实有能力移除或增加信息到cell状态中，由被称为门的结构精细控制。

门是一种让信息可选地通过的方法。它们由一个sigmoid神经网络层和一个点乘操作组成。

![](../../res/LSTM3-gate.png)

sigmod层输出[0, 1]区间内的数，描述了每个部分中应该通过的比例。输出0意味着“什么都不能通过”，而输出1意味着“让所有东西通过！”。

一个LSTM有这个这样的门，以保护和控制cell的状态。

## 深入浅出LSTM

我们的LSTM的第一步是决定我们需要从cell状态中扔掉什么样的信息。这个决策由一个称为“遗忘门”的sigmoid层做出。它
The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1ht−1 and xtxt, and outputs a number between 00 and 11 for each number in the cell state Ct−1Ct−1. A 11 represents “completely keep this” while a 00 represents “completely get rid of this.”

Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.


The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~tC~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.


It’s now time to update the old cell state, Ct−1Ct−1, into the new cell state CtCt. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by ftft, forgetting the things we decided to forget earlier. Then we add it∗C~tit∗C~t. This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.


Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanhtanh (to push the values to be between −1−1 and 11) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.


Variants on Long Short Term Memory

What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.

One popular LSTM variant, introduced by Gers & Schmidhuber (2000), is adding “peephole connections.” This means that we let the gate layers look at the cell state.


The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.


A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

A gated recurrent unit neural network.
These are only a few of the most notable LSTM variants. There are lots of others, like Depth Gated RNNs by Yao, et al. (2015). There’s also some completely different approach to tackling long-term dependencies, like Clockwork RNNs by Koutnik, et al. (2014).

Which of these variants is best? Do the differences matter? Greff, et al. (2015) do a nice comparison of popular variants, finding that they’re all about the same. Jozefowicz, et al. (2015) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

Conclusion

Earlier, I mentioned the remarkable results people are achieving with RNNs. Essentially all of these are achieved using LSTMs. They really work a lot better for most tasks!

Written down as a set of equations, LSTMs look pretty intimidating. Hopefully, walking through them step by step in this essay has made them a bit more approachable.

LSTMs were a big step in what we can accomplish with RNNs. It’s natural to wonder: is there another big step? A common opinion among researchers is: “Yes! There is a next step and it’s attention!” The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs. In fact, Xu, et al. (2015) do exactly this – it might be a fun starting point if you want to explore attention! There’s been a number of really exciting results using attention, and it seems like a lot more are around the corner…

Attention isn’t the only exciting thread in RNN research. For example, Grid LSTMs by Kalchbrenner, et al. (2015) seem extremely promising. Work using RNNs in generative models – such as Gregor, et al. (2015), Chung, et al. (2015), or Bayer & Osendorfer (2015) – also seems very interesting. The last few years have been an exciting time for recurrent neural networks, and the coming ones promise to only be more so!