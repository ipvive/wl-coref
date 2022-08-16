token attentions[...][512][512]

   cls,sep,pad removal (2x slice)

token_masked_attentions[...][nsw][nsw]

   out[x][y] = A * in[:,:,x,y] where A is trained

token_logits[nsw][nsw]

    mm by arange hack with softmax instead of log

word_logits[nw][nw]
   
   add eye(-inf)  (rough_scorer masking)

bilinear_scores[nw][nw]

   (inject instead of rough_scorer output)


Q: do we want dropout somewhere?
Q: is there a good way to add a trainable attention to entry word matrix?
