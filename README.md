# [Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling](https://www.techrxiv.org/users/867895/articles/1291580-translution-unifying-self-attention-and-convolution-for-adaptive-and-relative-modeling)

⚠️ *Please note that a full Translution Neural Network requires a large amount of GPU memory—beyond what most current devices can provide.
However, you can replace individual Self-Attention layers with Translution in Transformers, which may yield surprisingly performance improvements.*

## Code Index 
* Image (2D): • [Translution](https://github.com/hehefan/Translution/blob/main/ViT/models/translution.py)   • [α-Translution](https://github.com/hehefan/Translution/blob/main/ViT/models/alpha_translution.py)

* Language (1D): • [Translution](https://github.com/hehefan/Translution/blob/main/GPT/models/translution.py)   • [α-Translution](https://github.com/hehefan/Translution/blob/main/GPT/models/alpha_translution.py)

## Abstract

When modeling a given type of data, we consider it to involve two key aspects:  1) identifying relevant  elements (e.g., image pixels or textual words) to a central element, as in a convolutional receptive field, or to a query element, as in self-attention, and 2) encoding these tokens effectively. Self-attention can adaptively identify these elements but relies on absolute positional embedding for structural representation learning.  In contrast, convolution encodes elements in a relative manner, yet their fixed kernel size limits their ability to adaptively select the relevant elements. Translution unifies the adaptive identification capability of self-attention and the relative encoding advantage of convolution. 

<img src="https://github.com/hehefan/Translution/blob/main/imgs/Translution.png"  width="60%" />



## Related Repos
1. ViT: https://github.com/lucidrains/vit-pytorch
2. nanoGPT: https://github.com/karpathy/nanoGPT
