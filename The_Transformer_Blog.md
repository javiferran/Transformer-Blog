
<h1><center>The Transformer Blog: fairseq edition</center></h1>

The Transformer was presented in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) and introduced a new architecture for many NLP tasks. In this post we exhibit an explanation of the Transformer architecture on Neural Machine Translation focusing on the [fairseq](https://github.com/pytorch/fairseq) implementation. We believe this could be useful for researchers and developers starting out on this framework.

The blog is inspired by [The annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) and [Fairseq Transformer, BART](https://yinghaowang.xyz/technology/2020-03-14-FairseqTransformer.html) blogs.

# Model Architecture

The Transformer is based on a stack of encoders and another stack of decoders. The encoder maps an input sequence of tokens $\mathcal{X}=(token_{0},...,token_{src\text{_}len})$ to a sequence of continuous vector representations $encoder\_out = (encoder\text{_}out_0, ..., encoder\text{_}out_{src\_len})$. Given $encoder\_out$, the decoder then generates an output sequence $\mathcal{Y} = (output_0,...,output_{T})$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next token.

![test](https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/transformer_javifer.png?raw=true | width=10)

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/transformer_javifer.png?raw=true" width="50%" align="center"/>

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

The encoder (<code class="language-plaintext highlighter-rouge">TransformerEncoder</code>) is composed of a stack of $N=encoder\_layers$ identical layers.

The encoder recieves a list of tokens $\mathcal{X}=$<code class="language-plaintext highlighter-rouge">src_tokens</code>$=(token_{0},...,token_{src\_len})$ which are then converted to continuous vector representions <code class="language-plaintext highlighter-rouge">x = self.forward_embedding(src_tokens, token_embeddings)</code>, which is made of the sum of the (scaled) embedding lookup and the positional embedding: <code class="language-plaintext highlighter-rouge">x = self.embed_scale * self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)</code>.

From now on, let's consider $X^L$ as the $L$ encoder layer input sequence. $X^{1}$ refers then to the vectors representation of the input sequence tokens of the first layer, after computing <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> on <code class="language-plaintext highlighter-rouge">src_tokens</code>.

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/operations.png?raw=true" width="50%" align="center"/>

Note that although $X^L$ is represented in fairseq as a tensor of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, for the shake of simplicity, we take <code class="language-plaintext highlighter-rouge">batch=1</code> in the upcoming mathematical notation and just consider it as a <code class="language-plaintext highlighter-rouge">src_len x encoder_embed_dim</code> matrix.

$$
X^L = \begin{bmatrix}
x_{0}\\
\vdots\\
x_{src\_len}
\end{bmatrix}
$$

Where $x_{i} \in \mathbb{R}^{encoder\_embed\_dim}$.


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

* encoder_out: of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, the last layer encoder's embedding which, as we will see, is used by the Decoder. Note that is the same as $X^{N+1}$ when <code class="language-plaintext highlighter-rouge">batch=1</code>.
* encoder_padding_mask: of shape <code class="language-plaintext highlighter-rouge">batch x src_len</code>. Binary ByteTensor where padding elements are indicated by 1.
* encoder_embedding: of shape <code class="language-plaintext highlighter-rouge">src_len x batch x encoder_embed_dim</code>, the words (scaled) embedding lookup.
* encoder_states: of shape <code class="language-plaintext highlighter-rouge">list[src_len x batch x encoder_embed_dim]</code>, intermediate enocoder layer's output.


## Encoder Layer

The previous snipped of code shows a loop over the layers of the Encoder block, <code class="language-plaintext highlighter-rouge">for layer in self.layers</code>. This layer is implemented in fairseq in <code class="language-plaintext highlighter-rouge">class TransformerEncoderLayer(nn.Module)</code> inside [fairseq/modules/transformer_layer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py) and computes the following operations:

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/encoder_javifer.png?raw=true" width="20%" align="center"/>

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

$$\text{Feed Forward}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$




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

Each encoder layer input $X^L$, shown as <code class="language-plaintext highlighter-rouge">query</code> below since three identical copies are passed to the self-attention module, is multiplied by three weight matrices learned during the training process: $W^{Q}, W^{K}$ and $W^{V}$, obtaining $Q$, $K$ and $V$. Each row of this output matrices represents the query, key and value vectors of each token in the sequence, represented as $q$, $k$ and $v$ in the formulas that follow.


```python
    if self.self_attention:
      q = self.q_proj(query) # Q
      k = self.k_proj(query) # K
      v = self.v_proj(query) # V
    q *= self.scaling
```

The self-attention module does the following operation:

$$
\mathrm{softmax}(\frac{QK^\top}{\sqrt{d_{k}}})V
$$


```python
    attn_weights = torch.bmm(q, k.transpose(1, 2)) # QK^T multiplication
```

Given a token in the input, $i \in X^L$, it is passed to the self-attention function. Then, by means of dot products, scalar values (scores) are obtained between the query vector $q_{i} = iW^Q$ and every key vector of the input sequence $k_{j}$. The intuition is that this performs a similarity operation, similar queries and keys vectors will yield higher scores.

This scores represents how much attention is paid by the self-attention layer to other parts of the sequence when encoding $i$. By multiplying $q_{i}$ by the matrix $K^{T}$, a list of <code class="language-plaintext highlighter-rouge">src_len</code> scores is output. The scores are then passed through a softmax function giving bounded values:

$$
\alpha_{i} = \text{softmax}(\frac{\mathbf{q}_i {K}^\top}{\sqrt{d_k}})
= \frac{\exp(\frac{\mathbf{q}_i {K}^\top}{\sqrt{d_k}})}{ \sum_{j=0}^{src\_len} \exp(\frac{\mathbf{q}_i k_{j}^\top}{\sqrt{d_k}})}
$$


```python
    attn_weights_float = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )
    attn_weights = attn_weights_float.type_as(attn_weights)
```

The division by the square root of the dimension of the key vectors $d_{k}$ (for getting more stable gradients) is done previously <code class="language-plaintext highlighter-rouge">q *= self.scaling</code> instead in fairseq.


For example, given the sentence "the nice cat walks away from us" for the token $i=\text{from}$, its corresponding attention weights $\alpha_{i}$ for every other token $j$ in the input sequence could be:

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/probs.jpg?raw=true" style="width:600px;height:250px;" align="center"/>

Once we have normalized scores for every pair of tokens $\{i,j\}$, we multiply these weights by the value vector $v_{j} \forall j \in X^L$ (each row in matrix $V$) and finally sum up those vectors:

$$
z_{i} = \sum_{j=0}^{src\_len}\alpha_{i,j}v_{j}
$$

Where $z_{i}$ represents row $i$ of $Z$. By doing the matrix multiplication of the attention weight matrix <code class="language-plaintext highlighter-rouge">attn_weights</code> and $V$, $\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V$, we directly get matrix $Z$.


```python
    attn_probs = self.dropout_module(attn_weights)
    assert v is not None
    attn = torch.bmm(attn_probs, v)
```

This process is done in parallel in each of the self-attention heads. So, in total <code class="language-plaintext highlighter-rouge">encoder_attention_heads</code> matrices are output. Each head has its own $W^{Q}, W^{K}$ and $W^{V}$ weight matrices which are randomly initialized, so the result leads to different representation subspaces in each of the self-attention heads.

The output matrices $Z$ of every self-attention head are concatenated into a single one to which a linear transformation $W^{O}$ (<code class="language-plaintext highlighter-rouge">self.out_proj</code>) is applied, $$attn = Concat(Z^{head_{i}},\cdots,Z^{head_{h}})W^{O}$$


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


To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=$<code class="language-plaintext highlighter-rouge">encoder_embed_dim</code>.

# Decoder

The decoder is composed of a stack of $N=decoder\_layers$ identical layers.

The goal of the decoder is to generate a sequence $\mathcal{Y}$ in the target language. The <code class="language-plaintext highlighter-rouge">TransformerDecoder</code> inherits from <code class="language-plaintext highlighter-rouge">FairseqIncrementalDecoder</code>. It differs from the encoder in that it performs incremental decoding. This means that at each time step $t$ a forward pass is done through the decoder, generating $output_{t}$, which is then fed as input to the next time step decoding process.

The encoder output <code class="language-plaintext highlighter-rouge">encoder_out.encoder_out</code> is used by the decoder (in each layer) together with $\mathcal{Y<t}=(output_{0},...,output_{t-1})$ (<code class="language-plaintext highlighter-rouge">prev_output_tokens</code>) to generate one feature vector per target token at each time step (<code class="language-plaintext highlighter-rouge">tgt_len = 1</code> in each forward pass). This feature vector is then transformed by a linear layer and passed through a softmax layer <code class="language-plaintext highlighter-rouge">self.output_layer(x)</code> to get a probability distribution over the target language vocabulary.

Following the beam search algorithm, top <code class="language-plaintext highlighter-rouge">beam</code> hypothesis are chosen and inserted in the batch dimension input of the decoder (<code class="language-plaintext highlighter-rouge">prev_output_tokens</code>) for the next time step.

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/decoder_javifer.png?raw=true" style="width:400px;height:500px;" align="center"/>

We consider $query^L$ as the $L$ decoder layer input sequence. $query^{1}$ refers then to the vector representation of the input sequence tokens of the first layer, after computing <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> on <code class="language-plaintext highlighter-rouge">prev_output_tokens</code>. Note that here <code class="language-plaintext highlighter-rouge">self.forward_embedding</code> is not defined, but we refer to <code class="language-plaintext highlighter-rouge">self.embed_tokens(prev_output_tokens)</code> and <code class="language-plaintext highlighter-rouge">self.embed_positions(prev_output_tokens)</code>.


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

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/incremental_decoding.png?raw=true" style="width:450px;height:550px;" align="center"/>

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer (Encoder-Decoder Attention), which performs multi-head attention over the output of the encoder stack as input for $W^{K}$ and $W^{V}$ and the ouput of the sprevious module $attn_{t}*$.  Similar to the encoder, it employs residual connections around each of the sub-layers, followed by layer normalization.


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

During incremental decoding, $(output_{0},...,output_{t-2})$ enter the self-attention module as <code class="language-plaintext highlighter-rouge">prev_key</code> and <code class="language-plaintext highlighter-rouge">prev_value</code> vectors that are stored in <code class="language-plaintext highlighter-rouge">incremental_state</code>. Since there is no need to recompute $K$ and $V$ every time, incremental decoding caches these values and concatenates with keys an values from $output_{t-1}$. Then, updated $K$ and $V$ are stored in <code class="language-plaintext highlighter-rouge">prev_key</code> and passed again to <code class="language-plaintext highlighter-rouge">incremental_state</code>.

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/augmented_decoder_javifer_self_attn.png?raw=true" style="width:750px;height:350px;" align="center"/>

The last time step output token in each decoding step, $output_{t-1}$, enters as a query after been embedded. So, queries here have one element in the second dimension, that is, there is no need to use matrix $Q$ notation.
As before, scalar values (scores) $\alpha$ are obtained between the query vector $q_{t-1}$ and every key vector of the whole previous tokens sequence.

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

The Encoder-Decoder attention receives key and values from the encoder output <code class="language-plaintext highlighter-rouge">encoder_out.encoder_out</code> and the query from the previous module $attn_{t}*$. Here, $q_{t}$ is compared against every key vector received from the encoder (and transformed by $W^K$).

As before, $K$ and $V$ don't need to be recomputed every time step since they are constant for the whole decoding process. Encoder-Decoder attention uses <code class="language-plaintext highlighter-rouge">static_kv=True</code> so that there is no need to update the <code class="language-plaintext highlighter-rouge">incremental_state</code> (see previous code snipped).

Now, just one vector $z_{t}$ is generated at each time step by each head as a weighted average of the $v$ vectors.

<img src="https://github.com/javiferran/Transformer-Blog/blob/main/The_Transformer_Blog_files/augmented_decoder_javifer_enc_dec_attn.png?raw=true" style="width:600px;height:350px;" align="center"/>


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

$$\text{Feed Forward}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$



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
