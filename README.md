
<h1><center>The Transformer Blog: fairseq edition</center></h1>

The Transformer was presented in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) and introduced a new architecture for many NLP tasks. In this post we exhibit an explanation of the Transformer architecture on Neural Machine Translation focusing on the [fairseq](https://github.com/pytorch/fairseq) implementation. We believe this could be useful for researchers and developers starting out on this framework.

The blog is inspired by [The annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) and [Fairseq Transformer, BART](https://yinghaowang.xyz/technology/2020-03-14-FairseqTransformer.html) blogs.

# Model Architecture

The Transformer is based on a stack of encoders and another stack of decoders. The encoder maps an input sequence of tokens <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{X}=(token_{0},...,token_{src\_len})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\mathcal{X}=(token_{0},...,token_{src\_len})" title="\mathcal{X}=(token_{0},...,token_{src\_len})" /></a> to a sequence of continuous vector representations <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;encoder\_out&space;=&space;(encoder\_out_0,...,&space;encoder\_out_{src\_len})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;encoder\_out&space;=&space;(encoder\_out_0,...,&space;encoder\_out_{src\_len})" title="encoder\_out = (encoder\_out_0,..., encoder\_out_{src\_len})" /></a>. Given <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/e1f586bbabb07a23681373cd7beb57f8.svg?invert_in_darkmode" align=middle width=85.91816475pt height=22.8310566pt/>, the decoder then generates an output sequence <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/61223eff453f36f7c812cda03c623683.svg?invert_in_darkmode" align=middle width=186.9442245pt height=24.657534pt/> of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next token.<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

<p align="center">
    <img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/transformer_javifer.png?raw=true" width="50%" align="center"/>
</p>

To see the general structure of the code in fairseq implementation I recommend reading [Fairseq Transformer, BART](https://yinghaowang.xyz/technology/2020-03-14-FairseqTransformer.html).

This model is implemented in fairseq as <code class="language-plaintext highlighter-rouge">TransformerModel</code> in [fairseq/models/transformer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py).


```python
class TransformerModel(FairseqEncoderDecoderModel):
...
  def forward(
          self,
          src_tokens,
          src_lengths,
          prev_output_tokens,
          return_all_hiddens: bool = True,
          features_only: bool = False,
          alignment_layer: Optional[int] = None,
          alignment_heads: Optional[int] = None,
      ):
          encoder_out = self.encoder(
              src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
          )
          decoder_out = self.decoder(
              prev_output_tokens,
              encoder_out=encoder_out,
              features_only=features_only,
              alignment_layer=alignment_layer,
              alignment_heads=alignment_heads,
              src_lengths=src_lengths,
              return_all_hiddens=return_all_hiddens,
          )
          return decoder_out
```

# Encoder

The encoder (<code class="language-plaintext highlighter-rouge">TransformerEncoder</code>) is composed of a stack of <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/24d1e009effeb1a0bff15f63c766225d.svg?invert_in_darkmode" align=middle width=145.32061995pt height=22.8310566pt/> identical layers.

The encoder recieves a list of tokens <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/7e3b6af8eae728308fd3e221e88c9a63.svg?invert_in_darkmode" align=middle width=31.48399485pt height=22.4657235pt/><code class="language-plaintext highlighter-rouge">src_tokens</code><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/0268ff39fd834f42315ef1d7da9e86d6.svg?invert_in_darkmode" align=middle width=189.48573435pt height=24.657534pt/> which are then converted to continuous vector representions <code class="language-plaintext highlighter-rouge">x = self.forward_embedding(src_tokens, token_embeddings)</code>, which is made of the sum of the (scaled) embedding lookup and the positional embedding: <code class="language-plaintext highlighter-rouge">x = self.embed_scale * self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)</code>.

From now on, let's consider <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/b644454401a906aabdea6143648f30b0.svg?invert_in_darkmode" align=middle width=23.9269899pt height=27.6567522pt/> as the <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724255pt height=22.4657235pt/> encoder layer input sequence. <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/e52d0ca095f92c3fb610a9bcc324ff04.svg?invert_in_darkmode" align=middle width=21.4612134pt height=26.7617526pt/> refers then to the vectors representation of the input sequence tokens of the first layer, after computing <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> on <code class="language-plaintext highlighter-rouge">src_tokens</code>.

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/operations.png?raw=true" width="45%" align="center"/>
</p>

Note that although <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/b644454401a906aabdea6143648f30b0.svg?invert_in_darkmode" align=middle width=23.9269899pt height=27.6567522pt/> is represented in fairseq as a tensor of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, for the shake of simplicity, we take <code class="language-plaintext highlighter-rouge">batch=1</code> in the upcoming mathematical notation and just consider it as a <code class="language-plaintext highlighter-rouge">src_len x encoder_embed_dim</code> matrix.

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/1f3cf080d6a035537f35f735f85431f0.svg?invert_in_darkmode" align=middle width=120.6422019pt height=69.04177335pt/></p>

Where <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/1fb895c0276e1d1149b7e80b8bc0387b.svg?invert_in_darkmode" align=middle width=162.45174825pt height=27.9124395pt/>.


```python
class TransformerEncoder(FairseqEncoder):
...
  def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # batch x src_lengths x encoder_embed_dim
        #                     -> src_lengths x batch x encoder_embed_dim
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # src_lengths x batch x encoder_embed_dim
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=encoder_states, # List[src_lengths x batch x encoder_embed_dim]
            src_tokens=None,
            src_lengths=None,
        )
```

This returns a NamedTuple object <code class="language-plaintext highlighter-rouge">encoder_out</code>.

* encoder_out: of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, the last layer encoder's embedding which, as we will see, is used by the Decoder. Note that is the same as <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/885a53b1f01b4f642b4741264f1932b9.svg?invert_in_darkmode" align=middle width=43.1987259pt height=27.6567522pt/> when <code class="language-plaintext highlighter-rouge">batch=1</code>.
* encoder_padding_mask: of shape <code class="language-plaintext highlighter-rouge">batch x src_len</code>. Binary ByteTensor where padding elements are indicated by 1.
* encoder_embedding: of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, the words (scaled) embedding lookup.
* encoder_states: of shape <code class="language-plaintext highlighter-rouge">list[src_len x batch x encoder_embed_dim]</code>, intermediate enocoder layer's output.


## Encoder Layer

The previous snipped of code shows a loop over the layers of the Encoder block, <code class="language-plaintext highlighter-rouge">for layer in self.layers</code>. This layer is implemented in fairseq in <code class="language-plaintext highlighter-rouge">class TransformerEncoderLayer(nn.Module)</code> inside [fairseq/modules/transformer_layer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py) and computes the following operations:

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/encoder_javifer.png?raw=true" width="25%" align="center"/>
</p>

The input of the encoder layer is passed through the self-attention module <code class="language-plaintext highlighter-rouge">self.self_attn</code>, dropout (<code class="language-plaintext highlighter-rouge">self.dropout_module(x)</code>) is then applied before getting to the Residual & Normalization module (made of a residual connection <code class="language-plaintext highlighter-rouge">self.residual_connection(x, residual)</code> and a layer normalization (LayerNorm) <code class="language-plaintext highlighter-rouge">self.self_attn_layer_norm(x)</code>


```python
class TransformerEncoderLayer(nn.Module):
...
  def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
    if attn_mask is not None:
      attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

    residual = x
    if self.normalize_before:
        x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(
        query=x,
        key=x,
        value=x,
        key_padding_mask=encoder_padding_mask,
        attn_mask=attn_mask,
    )
    x = self.dropout_module(x)
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
        x = self.self_attn_layer_norm(x)
```

Then, the result is passed through a position-wise feed-forward network composed by two fully connected layers, <code class="language-plaintext highlighter-rouge">fc1</code> and <code class="language-plaintext highlighter-rouge">fc2</code> with a ReLU activation in between (<code class="language-plaintext highlighter-rouge">self.activation_fn(self.fc1(x))</code>) and dropout <code class="language-plaintext highlighter-rouge">self.dropout_module(x)</code>.

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/9313328dbc86098edf0b1d30cd604fdc.svg?invert_in_darkmode" align=middle width=324.22381695pt height=16.438356pt/></p>




```python
    residual = x
    if self.normalize_before:
        x = self.final_layer_norm(x)

    x = self.activation_fn(self.fc1(x))
    x = self.activation_dropout_module(x)
    x = self.fc2(x)
    x = self.dropout_module(x)
       
```

Finally, a residual connection is made before another layer normalization layer.


```python
    x = self.residual_connection(x, residual)
    if not self.normalize_before:
        x = self.final_layer_norm(x)
    return x
```

## Self-attention

As we have seen, the input of each encoder layer is firstly passed through a self-attention layer ([fairseq/modules/multihead_attention.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py))


```python
class MultiheadAttention(nn.Module):
...
  def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
```

Each encoder layer input <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/b644454401a906aabdea6143648f30b0.svg?invert_in_darkmode" align=middle width=23.9269899pt height=27.6567522pt/>, shown as <code class="language-plaintext highlighter-rouge">query</code> below since three identical copies are passed to the self-attention module, is multiplied by three weight matrices learned during the training process: <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/99786718f8a5ddffe56f5d617d1d94a0.svg?invert_in_darkmode" align=middle width=65.94849195pt height=27.6567522pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/c54c44baf0f59cf6d6b055c41bdbe1a9.svg?invert_in_darkmode" align=middle width=28.4019351pt height=27.6567522pt/>, obtaining <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.99542475pt height=22.4657235pt/>, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700595pt height=22.4657235pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/>. Each row of this output matrices represents the query, key and value vectors of each token in the sequence, represented as <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.92810645pt height=14.1552444pt/>, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.07536795pt height=22.8310566pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode" align=middle width=8.5578603pt height=14.1552444pt/> in the formulas that follow.


```python
    if self.self_attention:
      q = self.q_proj(query) # Q
      k = self.k_proj(query) # K
      v = self.v_proj(query) # V
    q *= self.scaling
```

The self-attention module does the following operation:

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/3a348f98715b48ac08043d1c566df088.svg?invert_in_darkmode" align=middle width=125.91348495pt height=40.36437405pt/></p>


```python
    attn_weights = torch.bmm(q, k.transpose(1, 2)) # QK^T multiplication
```

Given a token in the input, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/f5f2f2c855a45372a00bdeb61db7ac67.svg?invert_in_darkmode" align=middle width=49.6813548pt height=27.6567522pt/>, it is passed to the self-attention function. Then, by means of dot products, scalar values (scores) are obtained between the query vector <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/435807b66ef48b96ec9bc3fa440d4af0.svg?invert_in_darkmode" align=middle width=68.5527513pt height=27.6567522pt/> and every key vector of the input sequence <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/86dd2fb386abbf5644ffea6b170b2fa2.svg?invert_in_darkmode" align=middle width=14.6623851pt height=22.8310566pt/>. The intuition is that this performs a similarity operation, similar queries and keys vectors will yield higher scores.

This scores represents how much attention is paid by the self-attention layer to other parts of the sequence when encoding <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.6632257pt height=21.6830097pt/>. By multiplying <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/681f6e9a71298bc56353f18b64c61d09.svg?invert_in_darkmode" align=middle width=11.98921185pt height=14.1552444pt/> by the matrix <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/9e7b099e7a6229c71c9804a41aae5e67.svg?invert_in_darkmode" align=middle width=24.67071915pt height=27.6567522pt/>, a list of <code class="language-plaintext highlighter-rouge">src_len</code> scores is output. The scores are then passed through a softmax function giving bounded values:

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/5dec71fb60539bbe27c1c3da07944584.svg?invert_in_darkmode" align=middle width=311.57695635pt height=60.6040743pt/></p>


```python
    attn_weights_float = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )
    attn_weights = attn_weights_float.type_as(attn_weights)
```

The division by the square root of the dimension of the key vectors <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/861cd904487cbd9a710c0953f8d4de17.svg?invert_in_darkmode" align=middle width=15.82199355pt height=22.8310566pt/> (for getting more stable gradients) is done previously <code class="language-plaintext highlighter-rouge">q *= self.scaling</code> instead in fairseq.


For example, given the sentence "the nice cat walks away from us" for the token <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/0ff3d04662db5f0751a34d6234dbbd59.svg?invert_in_darkmode" align=middle width=60.95998095pt height=22.8310566pt/>, its corresponding attention weights <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/2e32e0141d372413f25c35045d246695.svg?invert_in_darkmode" align=middle width=15.1665459pt height=14.1552444pt/> for every other token <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710417pt height=21.6830097pt/> in the input sequence could be:

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/probs.jpg?raw=true" width="50%" align="center"/>
</p>

Once we have normalized scores for every pair of tokens <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/45cec34bd24a46fe74066f1fef04d815.svg?invert_in_darkmode" align=middle width=37.11794625pt height=24.657534pt/>, we multiply these weights by the value vector <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/c7170166449644ea182e87431e4ee7d4.svg?invert_in_darkmode" align=middle width=75.7554633pt height=27.6567522pt/> (each row in matrix <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/>) and finally sum up those vectors:

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/0f68df47e2c1115cb6faac7e3901d7bb.svg?invert_in_darkmode" align=middle width=119.685258pt height=50.1713685pt/></p>

Where <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/6ce0c310ecc16752f6226e50eaf576a0.svg?invert_in_darkmode" align=middle width=12.2955525pt height=14.1552444pt/> represents row <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.6632257pt height=21.6830097pt/> of <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397275pt height=22.4657235pt/>. By doing the matrix multiplication of the attention weight matrix <code class="language-plaintext highlighter-rouge">attn_weights</code> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/>, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/72f2853c9d5e6392be65a548b35a382d.svg?invert_in_darkmode" align=middle width=117.97907385pt height=35.8209489pt/>, we directly get matrix <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397275pt height=22.4657235pt/>.


```python
    attn_probs = self.dropout_module(attn_weights)
    assert v is not None
    attn = torch.bmm(attn_probs, v)
```

This process is done in parallel in each of the self-attention heads. So, in total <code class="language-plaintext highlighter-rouge">encoder_attention_heads</code> matrices are output. Each head has its own <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/99786718f8a5ddffe56f5d617d1d94a0.svg?invert_in_darkmode" align=middle width=65.94849195pt height=27.6567522pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/c54c44baf0f59cf6d6b055c41bdbe1a9.svg?invert_in_darkmode" align=middle width=28.4019351pt height=27.6567522pt/> weight matrices which are randomly initialized, so the result leads to different representation subspaces in each of the self-attention heads.

The output matrices <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397275pt height=22.4657235pt/> of every self-attention head are concatenated into a single one to which a linear transformation <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/8b85e13c718ff3f2005de8e9007cba05.svg?invert_in_darkmode" align=middle width=28.16078595pt height=27.6567522pt/> (<code class="language-plaintext highlighter-rouge">self.out_proj</code>) is applied, <p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/4f74c0544cc6a38253ca3032753d0fe9.svg?invert_in_darkmode" align=middle width=280.1420754pt height=18.88772655pt/></p>


```python
    #concatenating each head representation before W^o projection
    attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #W^o projection
    attn = self.out_proj(attn)
    attn_weights: Optional[Tensor] = None
    if need_weights:
        attn_weights = attn_weights_float.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=0)

    return attn, attn_weights
```

Notice that <code class="language-plaintext highlighter-rouge">attn_probs</code> has dimensions (bsz * self.num_heads, tgt_len, src_len)


To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/98a955c062800b5de9ee937236b2dac2.svg?invert_in_darkmode" align=middle width=61.26152505pt height=22.8310566pt/><code class="language-plaintext highlighter-rouge">encoder_embed_dim</code>.

# Decoder

The decoder is composed of a stack of <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/64a75ff298eb64b2cfa994a90e94dc02.svg?invert_in_darkmode" align=middle width=144.00970815pt height=22.8310566pt/> identical layers.

The goal of the decoder is to generate a sequence <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode" align=middle width=12.3379542pt height=22.4657235pt/> in the target language. The <code class="language-plaintext highlighter-rouge">TransformerDecoder</code> inherits from <code class="language-plaintext highlighter-rouge">FairseqIncrementalDecoder</code>. It differs from the encoder in that it performs incremental decoding. This means that at each time step <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.93609775pt height=20.2218027pt/> a forward pass is done through the decoder, generating <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/fe7a52162a2b7e2177956943ddba111d.svg?invert_in_darkmode" align=middle width=51.8971497pt height=20.2218027pt/>, which is then fed as input to the next time step decoding process.

The encoder output <code class="language-plaintext highlighter-rouge">encoder_out.encoder_out</code> is used by the decoder (in each layer) together with <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/41360a81c4ea155c215c5af6942b651e.svg?invert_in_darkmode" align=middle width=232.07941515pt height=24.657534pt/> (<code class="language-plaintext highlighter-rouge">prev_output_tokens</code>) to generate one feature vector per target token at each time step (<code class="language-plaintext highlighter-rouge">tgt_len = 1</code> in each forward pass). This feature vector is then transformed by a linear layer and passed through a softmax layer <code class="language-plaintext highlighter-rouge">self.output_layer(x)</code> to get a probability distribution over the target language vocabulary.

Following the beam search algorithm, top <code class="language-plaintext highlighter-rouge">beam</code> hypothesis are chosen and inserted in the batch dimension input of the decoder (<code class="language-plaintext highlighter-rouge">prev_output_tokens</code>) for the next time step.

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/decoder_javifer.png?raw=true" width="35%" align="center"/>
</p>

We consider <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/b112035aafe91cb9766b9dab30438464.svg?invert_in_darkmode" align=middle width=50.5329825pt height=27.6567522pt/> as the <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724255pt height=22.4657235pt/> decoder layer input sequence. <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/7bdea379fe614910a64d251ada5e3ce5.svg?invert_in_darkmode" align=middle width=48.067206pt height=26.7617526pt/> refers then to the vector representation of the input sequence tokens of the first layer, after computing <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> on <code class="language-plaintext highlighter-rouge">prev_output_tokens</code>. Note that here <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> is not defined, but we refer to <code class="language-plaintext highlighter-rouge">self.embed_tokens(prev_output_tokens)</code> and <code class="language-plaintext highlighter-rouge">self.embed_positions(prev_output_tokens)</code>.


```python
class TransformerDecoder(FairseqIncrementalDecoder):
...
  def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        
    x, extra = self.extract_features(
        prev_output_tokens,
        encoder_out=encoder_out,
        incremental_state=incremental_state,
        full_context_alignment=full_context_alignment,
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
    )
    if not features_only:
        x = self.output_layer(x)
    return x, extra
```


```python
def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
    return self.extract_features_scriptable(
        prev_output_tokens,
        encoder_out,
        incremental_state,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
    )
```

In the first time step, <code class="language-plaintext highlighter-rouge">prev_output_tokens</code> represents the beginning of sentence (BOS) token index. Its embedding enters the decoder as a tensor <code class="language-plaintext highlighter-rouge">beam*batch x tgt_len x encoder_embed_dim</code>.


```python
def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
  ..
    positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)
    if positions is not None:
            x += positions
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]
    for idx, layer in enumerate(self.layers):
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        x, layer_attn, _ = layer(
            x,
            encoder_out.encoder_out if encoder_out is not None else None,
            encoder_out.encoder_padding_mask if encoder_out is not None else None,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        if layer_attn is not None and idx == alignment_layer:
            attn = layer_attn.float().to(x)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if self.layer_norm is not None:
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        x = self.project_out_dim(x)

    return x, {"attn": [attn], "inner_states": inner_states}
```

## Decoder layer

The previous snipped of code shows a loop over the layers of the Decoder block <code class="language-plaintext highlighter-rouge">for idx, layer in enumerate(self.layers):</code>. This layer is implemented in fairseq in <code class="language-plaintext highlighter-rouge">class TransformerDecoderLayer(nn.Module)</code> inside [fairseq/modules/transformer_layer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py) and computes the following operations:

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/incremental_decoding.png?raw=true" width="40%" align="center"/>
</p>

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer (Encoder-Decoder Attention), which performs multi-head attention over the output of the encoder stack as input for <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/ae6f548aab05f80bedf7d706f1bb94c3.svg?invert_in_darkmode" align=middle width=29.6599248pt height=27.6567522pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/c54c44baf0f59cf6d6b055c41bdbe1a9.svg?invert_in_darkmode" align=middle width=28.4019351pt height=27.6567522pt/> and the ouput of the sprevious module <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/0aad597f864d22dfec0675cd38dd59ea.svg?invert_in_darkmode" align=middle width=44.43512205pt height=20.2218027pt/>.  Similar to the encoder, it employs residual connections around each of the sub-layers, followed by layer normalization.


```python
class TransformerDecoderLayer(nn.Module):
    ..
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):

        ...
        
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)      
        
```

## Self-attention in the decoder


```python
...

        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state, # previous keys and values stored here
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
```

During incremental decoding, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/cdcac3e959dd50a2245c79a138e45dba.svg?invert_in_darkmode" align=middle width=164.9472759pt height=24.657534pt/> enter the self-attention module as <code class="language-plaintext highlighter-rouge">prev_key</code> and <code class="language-plaintext highlighter-rouge">prev_value</code> vectors that are stored in <code class="language-plaintext highlighter-rouge">incremental_state</code>. Since there is no need to recompute <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700595pt height=22.4657235pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/> every time, incremental decoding caches these values and concatenates with keys an values from <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d0d2a05686fab7a15fde54ac0f114c22.svg?invert_in_darkmode" align=middle width=68.7237177pt height=20.2218027pt/>. Then, updated <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700595pt height=22.4657235pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/> are stored in <code class="language-plaintext highlighter-rouge">prev_key</code> and passed again to <code class="language-plaintext highlighter-rouge">incremental_state</code>.

<p align="center">
<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/augmented_decoder_javifer_self_attn.png?raw=true" width="65%" align="center"/>
</p>

The last time step output token in each decoding step, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d0d2a05686fab7a15fde54ac0f114c22.svg?invert_in_darkmode" align=middle width=68.7237177pt height=20.2218027pt/>, enters as a query after been embedded. So, queries here have one element in the second dimension, that is, there is no need to use matrix <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.99542475pt height=22.4657235pt/> notation.
As before, scalar values (scores) <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650495pt height=14.1552444pt/> are obtained between the query vector <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/89aafe4b9d21312d8427e0a2bcc390d9.svg?invert_in_darkmode" align=middle width=29.13067245pt height=14.1552444pt/> and every key vector of the whole previous tokens sequence.

Flashing back to ([fairseq/modules/multihead_attention.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py)) we can see how key and values are obtained inside the Multihead attention module and how these udates in  <code class="language-plaintext highlighter-rouge">saved_state</code> and  <code class="language-plaintext highlighter-rouge">incremental_state</code> are done:


```python
 class MultiheadAttention(nn.Module):
...
  def forward(
        ...
    ):
        ...
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state) # getting saved_state
        ...
        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv: # in encoder-endoder attention
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1) # concatenation of K
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv: # in encoder-endoder attention
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1) # concatenation of V
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state) # update
```

## Encoder-Decoder attention

The Encoder-Decoder attention receives key and values from the encoder output <code class="language-plaintext highlighter-rouge">encoder_out.encoder_out</code> and the query from the previous module <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/0aad597f864d22dfec0675cd38dd59ea.svg?invert_in_darkmode" align=middle width=44.43512205pt height=20.2218027pt/>. Here, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/88101fdb8c6c5c5b9ddad575a78144b7.svg?invert_in_darkmode" align=middle width=12.30410445pt height=14.1552444pt/> is compared against every key vector received from the encoder (and transformed by <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/6120b9f776c5da2bacb3c70cc8950569.svg?invert_in_darkmode" align=middle width=29.6599248pt height=27.6567522pt/>).

As before, <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700595pt height=22.4657235pt/> and <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.24203705pt height=22.4657235pt/> don't need to be recomputed every time step since they are constant for the whole decoding process. Encoder-Decoder attention uses <code class="language-plaintext highlighter-rouge">static_kv=True</code> so that there is no need to update the <code class="language-plaintext highlighter-rouge">incremental_state</code> (see previous code snipped).

Now, just one vector <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/2c96db0a1148e5d0dbbd2be27e9a871d.svg?invert_in_darkmode" align=middle width=12.6104451pt height=14.1552444pt/> is generated at each time step by each head as a weighted average of the <img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode" align=middle width=8.5578603pt height=14.1552444pt/> vectors.

<p align="center"><img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/augmented_decoder_javifer_enc_dec_attn.png?raw=true" width="45%" align="center"/>
</p>


```python
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            ...
            
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
```

As in the case of the encoder, the result is passed through a position-wise feed-forward network composed by two fully connected layers:

<p align="center"><img src="https://rawgit.com/javiferran/Transformer-Blog/main/svgs/9313328dbc86098edf0b1d30cd604fdc.svg?invert_in_darkmode" align=middle width=324.22381695pt height=16.438356pt/></p>



```python
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        
```

Finally, a residual connection is made before another layer normalization layer.


```python
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
...
        return x, attn, None
```
