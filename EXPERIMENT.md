 * njt proposed network
```
   (bert provided)

token attentions[...][512][512]

   cls,sep,pad removal (2x slice)

token_masked_attentions[...][nsw][nsw]

   out[x][y] = A * in[:,:,x,y] where A is trained (1x1 convolution)

token_logits[nsw][nsw]

    mm by arange hack with softmax instead of log

word_logits[nw][nw]
   
   add eye(-inf)  (rough_scorer masking)

bilinear_scores[nw][nw]

   (inject instead of rough_scorer output)
```

Q1: do we want dropout somewhere?
Q2: doo we need to add a trainable attention to entry word matrix?
    A: make sure it's broken before fixing Q1 then Q2.
