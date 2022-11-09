# GAN  

I used Gan using Generator, Swin Unet, and Discriminator, Resnet50.

At first, I will explain the Networks.

And After that, I  will tell how to use that code and modify the hyperparameters.

## Swin Transformer

There are many models to use as a generator. But we can recognize the fact that a variant of vision transformer models is on the top of ranks. Thus I want to use the "Swin Transformer" as a generator. 

But for anyone who want to use other models as a generator, I will affix other models in "Generators.py" 

First of all, I will explain the concept of vision transformer as a background knowledge of swin transformer and to understand what is the advancements of swin transformer.
  1. Vision Transformer
    
![vit_framework](https://user-images.githubusercontent.com/117700793/200583935-322c2d75-35b7-43ba-95e0-006e0f775aac.PNG)

    This image is the framework of the ViT.
    
    Let's see step by step
    
    step1. Patch 
    
![PatchSplitting](https://user-images.githubusercontent.com/117700793/200587023-957008db-bc71-43c0-9542-c7f9ea57e0d1.PNG)
    
    Patching is like this. One image data is split by 9 components.
    And We will use this one a kind of tokens after the process of embedding.
    
    Step2. Embedding
    
![PatchSpliting+Position_Embedding](https://user-images.githubusercontent.com/117700793/200587311-8ce4cbe1-77bd-4e0a-8d5e-93782cb0cbbd.PNG)
    
    Embedding is like this. After patching, Slope parameters is multiplied at each patches and after that the constant parameter is added on the result of the Linear projection. The final process of adding  is called of Position Embedding. Because this process is a kind of notification of image patch's position.
    
    Step3. Transformer Encoder
    
    Embedding output is input of transformer's input.
    
![transformer_encoder](https://user-images.githubusercontent.com/117700793/200589005-1ba0da5f-1ec9-4ccc-b76c-62b8c22b58ef.PNG)
    
    In this process, we focus on "Multi Head Attention"
      
      1. Standard Self Head Attention
      
![standard_self_head_attention](https://user-images.githubusercontent.com/117700793/200592040-243c0ccc-2d4a-4d86-b21e-04a46651e7b1.PNG)
      
      At encoding output, key, and query parameters are multiplied and that each ouputs is inner-producted and then normalized and then applied with soft-max function       after that, Value parameters are producted.
      
      2. Multi Head Attention.
      ![MHA](https://user-images.githubusercontent.com/117700793/200593317-7bbc600d-4315-486e-aeb7-6e426c4bf305.PNG)
      
      the Standard self head attentions are iterated multiple times and then concatenated.
      
    Step4. MLP
    
    This process is the making classification.
    
  
  2. Swin Transformer


### References :

[Swin transformer](https://arxiv.org/pdf/2103.14030v1.pdf)

[ViT](https://arxiv.org/pdf/2010.11929.pdf)

[Swin Unet](https://arxiv.org/pdf/2105.05537.pdf)

[Residual Network](https://arxiv.org/pdf/1512.03385.pdf)

[GAN](https://arxiv.org/pdf/1406.2661.pdf)

