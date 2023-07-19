# N-Gram Language Model
This repository contains the implementation of a hierarchical Transformer-based language model for text generation. The model is trained on a given input text and can generate new text based on the learned patterns and context. 
The input text in this case is 1001 Nights.

## Prerequisites
Python 3.x
PyTorch
Installation
Clone the repository:
```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## Install the required dependencies:

```
pip install torch

```
## Usage

* Prepare your input text file (input.txt) and place it in the project directory.
* Open the main.py file and modify the hyperparameters as needed.

## Run the script to train the language model:

```
python main.py

```

## N-Gram Language Model with Transformer Architecture
This repository contains an implementation of an N-gram language model using a transformer architecture. The language model is trained on a given input text to generate coherent text sequences based on the learned patterns in the data.

## Dataset
The input text is read from a file named "input.txt". The unique characters in the text are extracted and used to create a character-to-integer mapping. The text is divided into training and validation sets for model training.

## Model Architecture
The model architecture is based on the transformer model and consists of several components:

Head: Represents one head of self-attention. It computes attention scores and weighted aggregations of values.

MultiHeadAttention: Combines multiple heads of self-attention in parallel. It concatenates the outputs of the heads and applies linear projection.

CustomGELU: Customized GELU activation function.

FeedForward: A simple linear layer followed by a non-linearity.

Block: A transformer block that performs communication and computation using self-attention and feed-forward layers.

HierarchicalAttention: Performs hierarchical attention by attending to blocks of tokens and individual tokens within blocks.

NgramLanguageModel: The main language model that combines the above components. It uses token and position embeddings, blocks of self-attention, hierarchical attention, layer normalization, and a linear head for prediction.

Training Loop
The model is trained using the Adam optimizer. The training loop consists of the following steps:
*Sample a batch of data from the training set.
*Evaluate the loss and perform backpropagation to update the model parameters.
*Every eval_interval iterations, evaluate the loss on both the training and validation sets using the estimate_loss() function.
*Print the training and validation losses at regular intervals.

Hyperparameters
*batch_size: Number of independent sequences processed in parallel.
*block_size: Maximum context length for predictions.
*max_iters: Maximum number of training iterations.
*eval_interval: Interval for evaluating the loss on training and validation sets.
*learning_rate: Learning rate for the optimizer.
*device: Device to run the computations on (CPU or CUDA).
*eval_iters: Number of iterations for estimating the loss during evaluation.
*n_embd: Embedding dimension.
*n_head: Number of heads for self-attention.
*n_layer: Number of transformer blocks.
*dropout: Dropout rate for regularization.
*ngram: Size of the N-gram context for generating new tokens.
*block_num_heads: Number of heads used in the hierarchical attention mechanism for attending to blocks of tokens.
*block_head_size: Size of each head in the block-level attention.
*token_num_heads: Number of heads used in the hierarchical attention mechanism for attending to individual tokens within blocks.
*token_head_size: Size of each head in the token-level attention.

Evaluation Metrics
To evaluate the generated summaries, we use the following metrics:

ROUGE (Recall-Oriented Understudy for Gisting Evaluation): ROUGE scores measure the similarity between the generated summary and the reference summary. Specifically, we use ROUGE-1, ROUGE-2, and ROUGE-L scores. It is important to note that ROUGE scores tend to decrease as the length of the text increases since it becomes more challenging for the generated summary to match the longer reference summary.

ROUGE-1 Score: 0.33
ROUGE-2 Score: 0.12
ROUGE-L Score: 0.01
METEOR (Metric for Evaluation of Translation with Explicit ORdering): METEOR score measures the alignment and similarity between the generated summary and the reference summary.

METEOR Score: 0.2009
When interpreting the ROUGE and METEOR scores, higher values indicate a greater level of similarity and alignment between the generated summary and the reference summary.

These scores reflect the performance of the model. Achieving a ROUGE-1 score of 0.33 and ROUGE-2 score of 0.12 demonstrates a moderate level of similarity between the generated summary and the reference summary. The ROUGE-L score of 0.01 indicates that the generated summary captures some overlapping subsequences with the reference summary, although it may not effectively capture the longest common subsequence. Additionally, the METEOR score of 0.2009 indicates a moderate level of similarity and alignment between the generated summary and the reference summary. Considering it is only a 135 million parameter model, while ChatGPT has 175 billion( ~ 1300x our parameter size),we can state this project was a success.

Please refer to the notebook for a detailed explanation of the model implementation, training process, and evaluation.


## Example generated text:

>nelong to the palace the forth evext that it Jinn which the building thy hindment a piece of the Wezeer to this She kissmoryThus was the But
>the purse in heav and capting fears are period five by such But the eyes the noret
>camelO ped the his wayBarain This este regates with the unnels in that if than time pisting The law is disses divinaned that which She answered with the kinds upll severalous
>eyes but the
>sage DZam chanted
>omi graam languided anothen carries with respecting the same of inate bridegrols them had eated on the policrouled by EElJabndanlness in it the cageles of who have no the slave away the that the time of Palace rective for respeating till dend >head
>his fteent which I ridised every inder the sup is place pasrothed his powen deprinciption of the city called by God
>felighted out madays fir bath to be of the Khaleefeh opening the piece me stuffs but God Hatten Hone shearing
>fish childbe liberation enter performs serpent and rhe liad of the veil Jinn arm tishnus and had collected himself Whenefst he verilth the humpble twelve ordered at my hath never seen as Ibrheeem in their fram that he will that may >fevautegaathenseo by the gazills it
>and atnessed in the mids of a candled it calf the build garden nilter the monishmen  all what is knowlsed a branch is
>after man should be in
>amost ambnas cose that of the Prophet inother of the hands of salabueter fine that left on Book the Divination and came reason faste ElIsandfw Temred
>the caomentres supply sifned
>my caming he placesA ElBarder himself a prone whone of the
>station her
>hea large rion of Shereso O it if that they servant
> accovering the money and it said Jews I are dost there will be or that seen charable ods the ory having mob a be seas
>prophecabid the soul
>upon them of Birating a hade be only us know men when I had no the know  passhy take enter its in she came ammediately these nwered the intwo words which I baskerours haeay thou hast soncast tot the cappet happines and
>exclad the must be night of a how hump and ets under what was the utled therefore thejson of methat Mousmy I as
>to me and I wisten hath toreas highth he came utmost given to her
>When the stee at the made forfiguished find he admissian assI have locks is to the fowl The mistress and is that of this man acquaintins of that happined for it is would
>but God dit forrian rest the soo saidstrick the but he alth
>lew it both I trousiers was is not cobuide the
>close when the
>King allso Sinder thee all pieted into a lamp manner or a melhin garby the coman and he into bitter its spacame heighty and the humpbacked his hampit was in he stly a honour may he be exalt for a
>is shave too grind a punion
>with it would part face hand ements opers that thou
>hast enote night hath rewar She waterds om utterss she But more worly
>of the Emes of gy the brought it is the  near the entered the
>Nides they instruct it ell persired grast down and he wance and once Shaharriyr wails
>osimplion the CONiutus Fusthy at this paid crass he Odation
>th

## License
This project is licensed under the MIT License.

Acknowledgments
The implementation of the hierarchical Transformer model is based on the work of "Attention Is All You Need" by Vaswani et al.
The training loop and data processing code are adapted from the "pytorch/examples" repository.
Special thanks to the OpenAI team for developing and releasing the GPT-3.5 model, which served as the foundation for this language model.
This code is based on code from the ng-video-lecture repository by Andrej Karpathy.

