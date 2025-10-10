# [Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling](https://www.techrxiv.org/users/867895/articles/1291580-translution-unifying-self-attention-and-convolution-for-adaptive-and-relative-modeling)

![](https://github.com/hehefan/Translution/blob/main/imgs/unification.png)
When modeling a given type of data, we consider it to involve two key aspects:  1) identifying relevant  elements (e.g., image pixels or textual words) to a central element, as in a convolutional receptive field, or to a query element, as in self-attention, and 2) encoding these tokens effectively.  Self-attention can adaptively identify these elements but relies on absolute positional embedding for structural representation learning.  In contrast, convolution encodes elements in a relative manner, yet their fixed kernel size limits their ability to adaptively select the relevant elements. Translution unifies the adaptive identification capability of self-attention and the relative encoding advantage of convolution. 
![](https://github.com/hehefan/Translution/blob/main/imgs/Translution.png)


## Related Repos
1. ViT: https://github.com/lucidrains/vit-pytorch
2. nanoGPT: https://github.com/karpathy/nanoGPT
